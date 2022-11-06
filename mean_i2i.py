import os, glob, argparse, pdb
import numpy as np

import hparams


def stack_emb(style_list, emb_dim=None):
    """Stack all emotion embeddings"""
    stacked = np.array([]).astype('float32').reshape(0, emb_dim)

    for fpath in style_list:
        emb = np.load(fpath).reshape(1, emb_dim)
        stacked = np.concatenate((stacked, emb))
        
    return stacked

def get_distance_single(current_sample, other_stack):
    """Compute distance between current sample and all samples of other stack"""
    distance = np.mean(np.sqrt(np.sum((current_sample - other_stack) ** 2, axis=1)), axis=0)
    return distance

def get_distance(current_stack, other_stack):
    """Compute distances among samples of current stack and samples of other stack"""
    distances = np.array([get_distance_single(sample, other_stack) for sample in current_stack])
    return distances

def get_i2i(current_stack, inter_distance, intra_distance, eps=np.finfo(np.float32).eps):
    """I2I algorithm"""
    ratio = inter_distance / (eps + intra_distance)
    return current_stack[np.argmax(ratio)]

def i2i_neutral(emb_dir, out_dir, speakers, emotions):
    """This code assumes that setting farthest emotion as neutral for all emotion"""
    # I2I algorithm depends on the distribution of the data, thus, the general approach is set the farthest emotion as neutral.
    # This code assumes that setting farthest emotion as neutral.
    # Therefore, 'neutral' should be first in the emotions list.
 
    for spk in speakers:
        for emo in emotions:
            current_style_list = glob.glob(os.path.join(emb_dir, emo) + '_' + spk + '*')
            current_style = stack_emb(current_style_list, hparams.E)
            mean_name = 'mean_' + emo + '_' + spk
            i2i_name  = 'i2i_' + emo + '_' + spk
            
            if emo == 'neutral':
                mean_emb = np.mean(current_style, axis=0)
                np.save(os.path.join(out_dir, mean_name), mean_emb)
                far_style = current_style # For i2i

            else:
                if len(current_style) != 0:
#                    mean_emb = np.mean(current_style, axis=0)
#                    np.save(os.path.join(out_dir, mean_name), mean_emb)

                    inter_dist = get_distance(current_style, far_style)
                    intra_dist = get_distance(current_style, current_style)
                    i2i_emb = get_i2i(current_style, inter_dist, intra_dist)

                    np.save(os.path.join(out_dir, i2i_name), i2i_emb)

def i2i_all(emb_dir, out_dir, speakers, emotions):
    """This code is for obtain i2i embedding while considering all emotion for farthest emotion"""
    for spk in speakers:
        for emo in emotions:
            current_style_list = glob.glob(os.path.join(emb_dir, emo) + '_' + spk + '*')
            current_style = stack_emb(current_style_list, hparams.E)

            if len(current_style) != 0:
                if emo == 'anger' or emo =='angry':
                    ang_style = current_style
                elif emo == 'sadness' or emo =='sad':
                    sad_style = current_style
                elif emo == 'happy':
                    hap_style = current_style
                elif emo == 'neutral':
                    neu_style = current_style

        # Angry
        inter_dist = (get_distance(ang_style, hap_style) + get_distance(ang_style, sad_style) + get_distance(ang_style, neu_style)) / 3
        intra_dist = get_distance(ang_style, ang_style)

        i2i_emb = get_i2i(ang_style, inter_dist, intra_dist)
        
        np.save(os.path.join(out_dir, 'i2i_angry_' + spk + '.npy'), i2i_emb)

        # Happy
        inter_dist = (get_distance(hap_style, ang_style) + get_distance(hap_style, sad_style) + get_distance(hap_style, neu_style)) / 3
        intra_dist = get_distance(hap_style, hap_style)

        i2i_emb = get_i2i(hap_style, inter_dist, intra_dist)
        
        np.save(os.path.join(out_dir, 'i2i_happy_' + spk + '.npy'), i2i_emb)

        # Sad
        inter_dist = (get_distance(sad_style, ang_style) + get_distance(sad_style, hap_style) + get_distance(sad_style, neu_style)) / 3
        intra_dist = get_distance(sad_style, sad_style)

        i2i_emb = get_i2i(sad_style, inter_dist, intra_dist)
        
        np.save(os.path.join(out_dir, 'i2i_sad_' + spk + '.npy'), i2i_emb)

        # Neutral
        inter_dist = (get_distance(neu_style, ang_style) + get_distance(neu_style, hap_style) + get_distance(neu_style, sad_style)) / 3
        intra_dist = get_distance(neu_style, neu_style)

        i2i_emb = get_i2i(neu_style, inter_dist, intra_dist)
        
        np.save(os.path.join(out_dir, 'i2i_neutral_' + spk + '.npy'), i2i_emb)


def main(emb_dir, out_dir, mode): 
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    # Data-specific configuration
    speakers = ['m1', 'm2', 'f1', 'f2']
    emotions = ['neutral', 'anger', 'angry', 'sadness', 'sad', 'happy'] 

    if mode == 'neu':
        i2i_neutral(emb_dir, out_dir, speakers, emotions)
    elif mode == 'all':
        i2i_all(emb_dir, out_dir, speakers, emotions)

    print('Processing done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--emb_dir', type=str, default='emb_dir', help='emotion embeddings extracted from data')
    parser.add_argument('-o', '--out_dir', type=str, default='emb_mean_dir', help='mean or i2i embeddings')
    parser.add_argument('--mode', type=str, default='neu', help='neu, all')
    args = parser.parse_args()
    
    main(args.emb_dir, args.out_dir, args.mode)
