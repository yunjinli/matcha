import numpy as np
import torch

from third_party.dift.dift_sd import SDFeaturizer4Eval

if __name__ == '__main__':
    categories = [None]

    cats_spair = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'dog', 'horse',
                  'motorbike', 'person', 'pottedplant', 'sheep', 'train', 'tvmonitor']
    cats_pascal = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                   'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                   'tvmonitor']

    cats_ap10k = ['alouatta', 'antelope', 'argali sheep', 'beaver', 'bison', 'black bear', 'bobcat', 'brown bear',
                  'buffalo', 'cat', 'cheetah', 'chimpanzee', 'cow', 'deer', 'dog', 'elephant', 'fox', 'giraffe',
                  'gorilla', 'hamster', 'hippo', 'horse', 'jaguar', 'king cheetah', 'leopard', 'lion', 'marmot',
                  'monkey', 'moose', 'mouse', 'noisy night monkey', 'otter', 'panda', 'panther', 'pig', 'polar bear',
                  'rabbit', 'raccoon', 'rat', 'rhino', 'sheep', 'skunk', 'snow leopard', 'spider monkey', 'squirrel',
                  'tiger', 'uakari', 'weasel', 'wolf', 'zebra']

    # cats_co3d = ['apple', 'banana', 'baseballbat', 'baseballglove', 'bicycle', 'car', 'cup', 'donut', 'frisbee',
    #              'hotdog', 'kite', 'microwave',
    #              'parkingmeter', 'pizza', 'sandwich', 'skateboard', 'stopsign', 'toybus', 'toyplane', 'tv']

    cats_willow = ['duck', 'winebottle']

    for cat in cats_spair + cats_pascal + cats_ap10k + cats_willow:
        # for cat in cats_spair:
        if cat not in categories:
            categories.append(cat)

    print(f'Find {len(categories)} category')
    dift = SDFeaturizer4Eval(device='cuda')

    category_convert_dict = {
        'aeroplane': 'airplane',
        'motorbike': 'motorcycle',
        'pottedplant': 'potted plant',
        'tvmonitor': 'tv',
    }

    cat_embeds = {}
    for cat in categories[:2]:
        # if cat in category_convert_dict.keys():
        #     cat = category_convert_dict[cat]
        print('cat: ', cat)
        with torch.no_grad():
            embed = dift.encode_prompt(cat=cat, device='cuda')
        print(embed.shape)
        if cat is None:
            cat_embeds['None'] = embed[0].detach().cpu().numpy()
        else:
            cat_embeds[cat] = embed[0].detach().cpu().numpy()

    np.savez('outputs/sd_category_embedding', **cat_embeds)
