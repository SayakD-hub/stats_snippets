import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from scipy.special import erfcinv
import random
import seaborn as sns
from scipy.stats import norm
import glob
import seaborn as sns
import logging
from pathlib import Path
from scipy.ndimage import label as cc_label
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq

def keep_largest_nested_bins_laminar(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    order = np.lexsort((
        -df["end_frame_idx"].to_numpy(),
        df["start_frame_idx"].to_numpy(),
    ))
    end = df["end_frame_idx"].to_numpy()[order]
    bins = df["bin_size"].to_numpy()[order]

    dominated = np.zeros(len(order), dtype=bool)
    stack = []  # nested chain of sorted indices

    for j in range(len(order)):
        while stack and end[stack[-1]] < end[j]:
            stack.pop()  # no longer contains j

        for i in stack:
            if bins[i] < bins[j]:
                dominated[i] = True
            elif end[i] == end[j] and bins[i] > bins[j]:
                dominated[j] = True

        stack.append(j)

    return df.iloc[order[~dominated]].reset_index(drop=True)

def grouping(full_seg_df,GROUPING_THRESHOLD,FILTER_THRESHOLD,MIN_FRAMES,MAX_FRAMES):
    try:
        frame = full_seg_df["frame_idx"].to_numpy()
        odo = full_seg_df["odometer_tick"].to_numpy()
        prob = full_seg_df['id_corro_prob'].to_numpy(dtype=float)
        
        full_seg_filt_df = full_seg_df.query('id_corro_prob >= @GROUPING_THRESHOLD').reset_index(drop=True)
        
        # fig, ax1 = plt.subplots(1, 1, figsize=(12, 6), sharex=True)
        # sns.lineplot(data=full_seg_filt_df, x="frame_idx", y="id_corro_prob")
        
        # label groups by cont frames
        resets = full_seg_filt_df['frame_idx'].diff() != 1
        resets.iloc[0] = False
        full_seg_filt_df['group_id'] = resets.cumsum()
        
        # filter small groups:
        groups = full_seg_filt_df['group_id'].unique()
        group_count_df = full_seg_filt_df[['inspection_id','group_id']].groupby('group_id').count().reset_index()
        good_grps = group_count_df.query('inspection_id >= @MIN_FRAMES')['group_id'].unique()
        full_seg_filt_df = full_seg_filt_df.query('group_id in @good_grps')
        
        
        # Split groups longer than MAX_FRAMES
        full_seg_filt_df['sub_group'] = full_seg_filt_df.groupby('group_id').cumcount() // MAX_FRAMES
        # 3. Create a unique identifier by combining both columns
        full_seg_filt_df['final_group'] = full_seg_filt_df['group_id'].astype(str) + "_" + full_seg_filt_df['sub_group'].astype(str)
        # Optional: Convert the combined IDs back to clean, sequential integers (0, 1, 2...)
        full_seg_filt_df['final_group_id'] = pd.factorize(full_seg_filt_df['final_group'])[0]
        full_seg_filt_df.drop(columns = ['sub_group','final_group'],inplace = True)
        
        # label group by max prob in group and count the number of frames in that group and the sum of the probs
        prob_max_df = full_seg_filt_df[['id_corro_prob','final_group_id']].groupby('final_group_id').agg(['max','count','sum']).reset_index()
        prob_max_df.columns = ['_'.join(col) for col in prob_max_df.columns]
        prob_max_df.rename(columns={'final_group_id_':'final_group_id'},inplace = True)
        full_seg_filt_df = full_seg_filt_df.merge(prob_max_df,left_on = 'final_group_id', right_on = 'final_group_id').reset_index(drop = True)
        
        
        # begin and end of frames and odometer ticks
        loc_df = full_seg_filt_df[['frame_idx','odometer_tick','final_group_id']].groupby('final_group_id').agg(['max','min']).reset_index()
        loc_df.columns = ['_'.join(col) for col in loc_df.columns]
        loc_df.rename(columns={'final_group_id_':'final_group_id'},inplace = True)
        full_seg_filt_df = full_seg_filt_df.merge(loc_df,left_on = 'final_group_id', right_on = 'final_group_id').reset_index(drop = True)
    
        # drop group if group threshold is not met and number of frames are below min frames
        full_seg_filt_df = full_seg_filt_df.query('id_corro_prob_max >= @FILTER_THRESHOLD and id_corro_prob_count >= @MIN_FRAMES').reset_index(drop = True)
        
        full_seg_filt_df.rename(
            columns={
                 'frame_idx_min':'start_frame_idx',
                 'frame_idx_max':'end_frame_idx',
                 'odometer_tick_min':'start_odo_ticks',
                 'odometer_tick_max':'end_odo_ticks',
                 'id_corro_prob_max':'vol_conf',
                 'id_corro_prob_sum':'total_vol_prob',
                 'id_corro_prob_count':'length'},inplace = True)
        
        full_seg_filt_df.drop(columns=['inspection_id','group_id','frame_idx', 'timer_tick','odometer_tick'], inplace=True)
        full_seg_filt_df.drop_duplicates(subset=['final_group_id'], inplace=True)
        
        full_seg_filt_df = full_seg_filt_df[['clip_id','track_id','final_group_id','start_odo_ticks', 'end_odo_ticks', 'start_frame_idx',
               'end_frame_idx', 'vol_conf','total_vol_prob', 'length']]
        return full_seg_filt_df
    except:
        return pd.DataFrame(columns=['clip_id','track_id','final_group_id','start_odo_ticks', 'end_odo_ticks', 'start_frame_idx',
               'end_frame_idx', 'vol_conf','total_vol_prob', 'length'])

def overlap_check(df: pd.DataFrame) -> bool:
    start = df["start_frame_idx"].to_numpy()
    end = df["end_frame_idx"].to_numpy()
    bins = df["bin_size"].to_numpy()
    #  this is a nested containment matrix that marks every point where an interval is totally contained in another
    # N^2 operations so do not use with large matrices
    contained = (start[:, None] <= start[None, :]) & (end[None, :] <= end[:, None])
    arr = np.array(contained)
    # Create a mask where diagonal elements are False and non-diagonal are True
    non_diag_mask = ~np.eye(arr.shape[0], arr.shape[1], dtype=bool)
    non_diag_bool = np.any(arr[non_diag_mask])
    return non_diag_bool


def prob_binning(bin_label,confidence_level,error_margin_prob):
    # designed for a two sided CI
    alpha = 1 - confidence_level
    z_score = norm.ppf(1 - alpha / 2)
    sample_size = int(np.ceil((z_score**2.0)*bin_label*(1-bin_label)/error_margin_prob**2.0))
    return sample_size
    


######## Main #######################################################

# now we loop over every single parquet on a single track 
# 1. Gather all the parquets in the track
# 2. For each parquet apply the grouping at various thresholds
# 3. Follow this by the cascade
# 4. concat the cascaded thresholds into one dataframe for the whole track

files_track_0 = glob.glob('/home/sayakdg/ILIWorkflows/workflows/corrosion_detection/data-files/0AXVHCA77S1/corrosion_detection/corrosion_v4.2/*/*/0/*parquet')
fullRawDF = pd.DataFrame()
fullstackDF = pd.DataFrame()
fullStackFiltDF = pd.DataFrame()
for file in tqdm(files_track_0):
    fileDF = pd.read_parquet(file)
    bins = np.arange(0.1,0.8,0.1)
    stackDF = pd.DataFrame()
    for bi in bins:
        df = grouping(fileDF,bi,bi,5,1000)
        if len(df) < 1:
            break
        else:
            df['bin_size'] = bi
            stackDF = pd.concat([stackDF,df]).reset_index(drop = True)
    stackFiltDF = keep_largest_nested_bins_laminar(stackDF)
    fullstackDF = pd.concat([stackDF,fullstackDF]).reset_index(drop = True)
    fullStackFiltDF = pd.concat([stackFiltDF,fullStackFiltDF]).reset_index(drop = True)
    fullRawDF = pd.concat([fullRawDF,fileDF.query('id_corro_prob >= 0.1').reset_index(drop = True)])

# add anomaly type column
fullStackFiltDF['anomaly_type'] = "ID_CORRO_POS"

# check if there are overlaps in identified frame ranges within a clip 
unique_clips = fullStackFiltDF['clip_id'].unique().to_numpy()
print('Unique clips present in the inspection:',unique_clips)
for clip in unique_clips:
    df = fullStackFiltDF.query('clip_id == @clip')
    print('For clip ',clip,' are there any overlapping segments:',overlap_check(df))


# write parquet for grouping out
table = pa.Table.from_pandas(fullStackFiltDF)
with pq.ParquetWriter('lower_conf_thresh_0.1_0.1_0.8.parquet', table.schema) as writer:
    writer.write_table(table)

# subsample each bin based on labeled probabilities
fullStackFiltDF['avg_defect_prob'] = fullStackFiltDF['total_vol_prob']/fullStackFiltDF['length']
bins = fullStackFiltDF['bin_size'].unique()
subSampleFullStackFiltDF = pd.DataFrame()
for bin_size in tqdm(bins):
    df = subSampleFullStackFiltDF.query('bin_size == @bin_size')
    bin_count = len(df)
    sampling_size = prob_binning(bin_size,0.95,0.1)
    print('Bin and corresponding sample size:',bin_size,sampling_size)
    if bin_count >= sampling_size:
        sampled_df = df.sample(n=sampling_size, weights='avg_defect_prob')
    else:
        sampled_df = df.sample(n=bin_count)
    subSampleFullStackFiltDF = pd.concat([subSampleFullStackFiltDF,sampled_df]).reset_index(drop = True)

subSampleFullStackFiltDF['anomaly_type'] = "ID_CORRO_POS"
subSampleFullStackFiltDF = subSampleFullStackFiltDF[['clip_id', 'start_odo_ticks', 'end_odo_ticks', 'start_frame_idx','end_frame_idx', 'vol_conf']]

table = pa.Table.from_pandas(subSampleFullStackFiltDF)
with pq.ParquetWriter('sub_sampled_lower_conf_thresh_0.1_0.1_0.8.parquet', table.schema) as writer:
    writer.write_table(table)

