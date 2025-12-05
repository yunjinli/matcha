import os
from copy import deepcopy
import torch
import numpy as np
import gradio as gr
import cv2
import gc

from matcha.feature.matcha_feature import MatchaFeature
from matcha.matcher.base_matcher import BaseMatcher
from matcha.benchmark.visualization import resize_img


class MatchingDemo:
    def __init__(self, matcher: BaseMatcher, device: torch.device, output_dir: str):
        self.matcher = matcher
        self.device = device

        self.source_image = None
        self.target_image = None
        self.source_feature = None
        self.target_feature = None

        self.examples = None
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.source_image = None
        self.target_image = None

        # Examples
        self.examples = [
            ["assets/examples/sacre_coeur_A.png", "assets/examples/sacre_coeur_B.png"],
            ["assets/examples/car_A.png", "assets/examples/car_B.png"],
            ["assets/examples/cat_A.png", "assets/examples/cat_B.png"],
            ["assets/examples/sheep_A.png", "assets/examples/sheep_B.png"],
            ["assets/examples/cat_A.png", "assets/examples/sheep_B.png"],
            ["assets/examples/horsejump-high_0.png", "assets/examples/horsejump-high_30.png"],
            ["assets/examples/chair_A.png", "assets/examples/chair_B.png"],
        ]

    def run(self):
        theme = gr.themes.Ocean()
        theme.set(
            checkbox_label_background_fill_selected="*button_primary_background_fill",
            checkbox_label_text_color_selected="*button_primary_text_color",
        )
        with gr.Blocks(theme=theme) as demo:
            gr.HTML(
                """
            <h1>MATCHA: Towards Matching Anything (CVPR 2025 Highlight)</h1>
            <p>
            <a href="https://github.com/feixue94/matcha">Source Code</a> |
            <a href="https://feixue94.github.io/matcha-project/">Project Page</a>
            </p>

            <div style="font-size: 18px; line-height: 1.5;">
            <p>Upload two images and click any point on the source image. MATCHA visualizes the most similar point and the normalized distance distribution on the target image.</p>
            
            </div>
            """
            )
            with gr.Row():
                source_img = gr.Image(label="Source Image")
                target_img = gr.Image(label="Target Image")

            source_img.upload(fn=self.upload_source_image, inputs=[source_img], outputs=[source_img])
            target_img.upload(fn=self.upload_target_image, inputs=[target_img], outputs=[target_img])
            source_img.select(
                fn=self.update_matching_point,
                inputs=[source_img, target_img],
                outputs=[source_img, target_img]
            )

            gr.Markdown("Click any row to load an example and wait for feature extraction.")
            gr.Examples(
                examples=self.examples,
                fn=self.process_example,
                inputs=[source_img, target_img],
                outputs=[source_img, target_img],
                cache_examples=False,
                run_on_click=True,  # must be True to call fn
            )

            demo.queue(max_size=20).launch(show_error=True, share=True)

    def upload_source_image(self, image):
        image = resize_img(image, nh=512)
        self.source_image = deepcopy(image)  # [h, w, 3]
        with torch.no_grad():
            img_rs = cv2.resize(image, (512, 512), interpolation=cv2.INTER_CUBIC)
            img_tensor = torch.from_numpy(img_rs / 255).float().permute(2, 0, 1)
            source_feature = self.matcher.describe(
                img=img_tensor[None].to(self.device), semantic_mode=True)
            self.source_feature = source_feature[0]

        return image

    def upload_target_image(self, image):
        image = resize_img(image, nh=512)
        self.target_image = deepcopy(image)
        with torch.no_grad():
            img_rs = cv2.resize(image, (512, 512), interpolation=cv2.INTER_CUBIC)
            img_tensor = torch.from_numpy(img_rs / 255).float().permute(2, 0, 1)
            target_feature = self.matcher.describe(
                img=img_tensor[None].to(self.device), semantic_mode=True)
            self.target_feature = target_feature[0]

            # self.target_feature = \
            #     torch.nn.Upsample(size=(image.shape[0], image.shape[1]), mode='bilinear')(target_feature)[0]
            # self.target_feature = F.normalize(self.target_feature, dim=1, p=2)[0]

        return image

    def update_matching_point(self, src_img, trg_img, evt: gr.SelectData, radius=15, alpha: float = 0.35):
        src_y, src_x = evt.index[1], evt.index[0]

        src_img_out = deepcopy(self.source_image)
        src_img_out = cv2.circle(
            src_img_out,
            center=(src_x, src_y),
            radius=radius,
            color=(255, 0, 0),
            thickness=-1
        )

        src_img_h, src_img_w = self.source_image.shape[:2]
        src_ft_h, src_ft_w = self.source_feature.shape[1:]

        src_vec = self.source_feature[:, int(src_y * src_ft_h / src_img_h), int(src_x * src_ft_w / src_img_w)]
        c = self.target_feature.shape[0]
        trg_vec = self.target_feature.view(c, -1)
        gc.collect()
        torch.cuda.empty_cache()

        trg_img_h, trg_img_w = self.target_image.shape[:2]
        trg_ft_h, trg_ft_w = self.target_feature.shape[1:]

        cos_map = torch.matmul(src_vec, trg_vec).view(trg_ft_h, trg_ft_w).cpu().numpy()  # H, W
        max_yx = np.unravel_index(cos_map.argmax(), cos_map.shape)

        heatmap = (cos_map - np.min(cos_map)) / (np.max(cos_map) - np.min(cos_map))  # normalize to [0, 1]
        heatmap = 1 - heatmap
        heatmap = (heatmap * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, dsize=(trg_img_w, trg_img_h), interpolation=cv2.INTER_CUBIC)
        trg_img_out = deepcopy(self.target_image)
        trg_img_out = (trg_img_out * (1 - alpha) + heatmap * alpha).astype(np.uint8)
        trg_img_out = cv2.circle(trg_img_out,
                                 center=(int(max_yx[1] * trg_img_w / trg_ft_w),
                                         int(max_yx[0] * trg_img_h / trg_ft_h)),
                                 radius=radius,
                                 color=(255, 0, 0),
                                 thickness=-1)  # fill
        return src_img_out, trg_img_out

    def process_example(self, source_image: np.ndarray, target_image: np.ndarray):
        self.source_image = deepcopy(source_image)  # [h, w, 3]
        with torch.no_grad():
            img_rs = cv2.resize(source_image, (512, 512), interpolation=cv2.INTER_CUBIC)
            img_tensor = torch.from_numpy(img_rs / 255).float().permute(2, 0, 1)
            source_feature = self.matcher.model.describe(
                img=img_tensor[None].to(self.device), semantic_mode=True)
        self.source_feature = source_feature[0]

        self.target_image = deepcopy(target_image)
        with torch.no_grad():
            img_rs = cv2.resize(target_image, (512, 512), interpolation=cv2.INTER_CUBIC)
            img_tensor = torch.from_numpy(img_rs / 255).float().permute(2, 0, 1)
            target_feature = self.matcher.model.describe(
                img=img_tensor[None].to(self.device), semantic_mode=True)
        self.target_feature = target_feature[0]

        # Return the images, so that the feature extraction can be done
        return source_image, target_image


if __name__ == "__main__":
    pretrained_path = "weights/matcha_pretrained.pth"
    model = MatchaFeature(config={"keypoint_method": None, "image_size": (512, 512)})
    model.load_state_dict(torch.load(pretrained_path), strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    matcher = BaseMatcher(model, device)

    demo = MatchingDemo(matcher=matcher, device=device, output_dir="outputs/demo")
    demo.run()

"""
1. load two images or a video (later)
2. show source and target image
3. select pixels from the source image 
4. click button compute to find retrieval points on the target image
5. re-visualize target image
"""
