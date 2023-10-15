import numpy as np
import pandas as pd
import time
import statsmodels.api as sm
from tqdm import tqdm
from pandarallel import pandarallel
import gc


import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

pandarallel.initialize(nb_workers=32)

def Find_Conditional_Factor(spot_data_dict, future_data_dict, mask_spot_data, mask_future_data, 
                            horizon_used, idx_list=[[2,0,0,1], [4,4,2,1]]):
    """
    This program mainly studies how to group targets according to Field B, and then calculates factors 
    in different ways within the group for Field A. The final factor expression is 
    < calculation_method_i(A_i) | B_i <- B_i comes from classification(B) >.
    
    Parameters:
        spot_data_dict: dict
            Input data for spot.
        future_data_dict: dict
            Input data for future.
        mask_spot_data: array-like
            Mask data for spot.
        mask_future_data: array-like
            Mask data for future.
        horizon_used: int
            Horizon value used for computations.
        idx_list: list of lists
            Nested list containing all required factor combinations.
            
    Returns:
        dict: Dictionary of computed features.
    """

    def get_A(A_id):
        """Fetch the A dataframe based on its ID."""

        if A_id == 2:
            """
            Calculates the intercept after regressing the incremental trading volume against minute returns. 

            Returns:
                ndarray: 3D array with shape (timestamp // bar, horizon_used, 195) containing the computed values.
            """

            # hyperparameters
            horizon_used = int(1440)
            bar = int(60)
            regression_lookback_length = int(10)

            # Extract relevant data
            close = future_data_dict['close'].values
            trade_size = future_data_dict['trade_size'].values
            shipan_mask = mask_future_data.values.astype(int)
            num_min, num_crypto = close.shape

            # Splitting the full data into 90-day training chunks
            chunk_size = 1440 * 30 * 3
            full_timestamps = np.arange(num_min)
            total_length = num_min
            split_times = total_length // chunk_size
            remainder = total_length % chunk_size
            training_indices = []

            for i in range(split_times):
                start_idx = i * chunk_size - horizon_used if i > 0 else 0
                end_idx = (i + 1) * chunk_size
                training_indices.append(full_timestamps[start_idx:end_idx])

            # If there's any data less than 90 days, consider them separately
            if remainder != 0:
                training_indices.append(full_timestamps[split_times * chunk_size - horizon_used:])

            full_Intercepts = np.empty((horizon_used // 60, num_crypto, horizon_used // 60))
            padded_value = np.full((horizon_used, num_crypto), -7)
            
            # If there's any data less than 90 days, consider them separately
            for indices in training_indices:
                print(f'Current Training Range: {pd.to_datetime(mask_future_data.index[indices[0]], unit="ms")} to {pd.to_datetime(mask_future_data.index[indices[-1]], unit="ms")}')
                
                # Padding data for further calculations
                close_extend = np.concatenate([padded_value, close[indices]], axis=0)
                trade_size_extend = np.concatenate([padded_value, trade_size[indices]], axis=0)

                # Creating sliding window views
                close_sliding_window = np.lib.stride_tricks.sliding_window_view(close_extend, horizon_used + 1, axis=0)[:,:,:-1]
                close_sliding_window = close_sliding_window[::bar] # (timestamps // 60, 195, horizon_used)
                trade_size_sliding_window = np.lib.stride_tricks.sliding_window_view(trade_size_extend, horizon_used + 1, axis=0)[:,:,:-1]
                trade_size_sliding_window = trade_size_sliding_window[::bar]

                del close_extend, trade_size_extend
                gc.collect()

                # Formatting and reshaping
                close_sliding_window_4d = close_sliding_window.reshape((close_sliding_window.shape[0], close_sliding_window.shape[1], horizon_used // 60, 60)) # (timestamp // bar, 195, 24, 60)
                trade_size_sliding_window_4d = trade_size_sliding_window.reshape((trade_size_sliding_window.shape[0], trade_size_sliding_window.shape[1], horizon_used // 60, 60))

                shift_close_sliding_window_4d =  close_sliding_window_4d.copy() # (timestamp // bar, 195, 24, 60)
                shift_close_sliding_window_4d[:,:,:,1:] = shift_close_sliding_window_4d[:,:,:,:-1]
                shift_close_sliding_window_4d[:,:,:,0] = np.nan
                return_sliding_window_4d = close_sliding_window_4d / (shift_close_sliding_window_4d + 1e-8) - 1

                del close_sliding_window_4d
                gc.collect()

                shift_trade_size_sliding_window_4d = trade_size_sliding_window_4d.copy() # (timestamp // bar, 195, 24, 60)
                shift_trade_size_sliding_window_4d[:,:,:,1:] = shift_trade_size_sliding_window_4d[:,:,:,:-1]
                shift_trade_size_sliding_window_4d[:,:,:,0] = np.nan
                trade_size_increase_sliding_window_4d = trade_size_sliding_window_4d - shift_trade_size_sliding_window_4d

                del trade_size_sliding_window_4d
                gc.collect()

                # Regression Target
                Target_sliding_window_4d = return_sliding_window_4d[:,:,:,regression_lookback_length+1:] # (timestamp // bar, 195, 24, 49)
                Y = Target_sliding_window_4d[:,:,:,:, np.newaxis] # (timestamp // bar, 195, 24, 49, 1)
                Y = np.nan_to_num(Y)
                trade_size_increase_sliding_window_4d = trade_size_increase_sliding_window_4d[:,:,:,1:]
                X = np.lib.stride_tricks.sliding_window_view(trade_size_increase_sliding_window_4d, regression_lookback_length+1, axis=-1) # (timestamp // bar, 195, 24, 49, 11)

                Ones = np.ones((X.shape[0], X.shape[1], X.shape[2], X.shape[3], 1))
                X = np.concatenate([Ones, X], axis=-1) # (timestamp // bar, 195, 24, 49, 12)
                X = np.nan_to_num(X)
                try:
                    X2 = np.linalg.pinv(X)
                except np.linalg.LinAlgError as e:
                    print("An error occurred:", str(e))
                    print("Matrix X:")
                    print(X)
                X3 = np.matmul(X2, Y) # (timestamp // bar, 195, 24, 12, 1)
                Intercepts_3d = X3.squeeze(-1)[:,:,:,0] # (timestamp // bar, 195, 24)

                del Target_sliding_window_4d, return_sliding_window_4d
                gc.collect()

                print(f'Current Timestamps Implemented: {pd.to_datetime(mask_future_data.index[indices[horizon_used::bar][0]], unit="ms")} \
                      to {pd.to_datetime(mask_future_data.index[indices[horizon_used::bar][-1]], unit="ms")}')
                full_Intercepts = np.concatenate([full_Intercepts, Intercepts_3d[horizon_used // 60:, :, :]], axis=0)

            full_Intercepts = full_Intercepts.transpose(0,2,1) # (timestamp // bar, 24, 195)

            return full_Intercepts

        if A_id == 4:
            """
            Computes the W slicing factor defined by the close/open prices.

            Returns:
            - ndarray: 3D array with shape (timestamp // bar, horizon_used, 195) containing the computed W factor values.
            """

            horizon_used = int(1440)
            bar = int(60)
            groupby = 60

            close = future_data_dict['close'].values
            open = future_data_dict['open'].values
            num_min, num_crypto = close.shape

            # Padding data for further calculations
            padded_value = np.full((horizon_used, num_crypto), -7)
            close_extend = np.concatenate([padded_value, close], axis=0)
            open_extend = np.concatenate([padded_value, open], axis=0)
            del padded_value
            gc.collect()

            # Creating sliding window views for close and open values
            close_sliding_window = np.lib.stride_tricks.sliding_window_view(close_extend, horizon_used + 1, axis=0)[:,:,:-1]
            close_sliding_window = close_sliding_window[::bar] # (timestamps // 60, 195, horizon_used)
            open_sliding_window = np.lib.stride_tricks.sliding_window_view(open_extend, horizon_used + 1, axis=0)[:,:,:-1]
            open_sliding_window = open_sliding_window[::bar]

            del close_extend, open_extend
            gc.collect()

            # Reshaping close and open sliding windows
            close_sliding_window_4d = close_sliding_window.reshape((close_sliding_window.shape[0], close_sliding_window.shape[1], horizon_used // groupby, groupby)) # (timestamp // bar, 195, horizon_used // groupby, groupby)
            open_sliding_window_4d = open_sliding_window.reshape((open_sliding_window.shape[0], open_sliding_window.shape[1], horizon_used // groupby, groupby))
            
            # Calculating the group return factor
            group_return_sliding_window_4d = close_sliding_window_4d[:,:,:,-1] / open_sliding_window_4d[:,:,:,0] - 1 # (timestamp // bar, 195, horizon_used // groupby)
            group_return_sliding_window_4d = group_return_sliding_window_4d.transpose(0,2,1)

            return group_return_sliding_window_4d # (timestamp // bar, horizon_used, 195)

    def get_B(B_id):
        """Fetch the B dataframe based on its ID."""

        if B_id == 0:
            "Dummy B"
            return None

        if B_id == 4:
            """
            Calculate the average trade value
            return:  3d array (timestamp, 195, horizon_used)
            """
            horizon_used = 1440
            bar = 60
            groupby = 60

            trade_value = future_data_dict['trade_value'].values
            trade_count = future_data_dict['trade_count'].values

            _, num_crypto = trade_value.shape
            padded_value = np.full((horizon_used, num_crypto), -7)
            
            trade_value_extend = np.concatenate([padded_value, trade_value], axis=0)
            trade_count_extend = np.concatenate([padded_value, trade_count], axis=0)

            del padded_value
            gc.collect()

            trade_value_sliding_window = np.lib.stride_tricks.sliding_window_view(trade_value_extend, horizon_used + 1, axis=0)[:,:,:-1] 
            trade_value_sliding_window = trade_value_sliding_window[::bar] # (timestamps // 60, 195, 1440)

            trade_count_sliding_window = np.lib.stride_tricks.sliding_window_view(trade_count_extend, horizon_used + 1, axis=0)[:,:,:-1]
            trade_count_sliding_window = trade_count_sliding_window[::bar]

            del trade_value_extend, trade_count_extend
            gc.collect()

            trade_value_sliding_window_4d = trade_value_sliding_window.reshape((trade_value_sliding_window.shape[0], trade_value_sliding_window.shape[1], horizon_used // groupby, groupby)) # (timestamp // bar, 195, 24, 60)
            trade_count_sliding_window_4d = trade_count_sliding_window.reshape((trade_count_sliding_window.shape[0], trade_count_sliding_window.shape[1], horizon_used // groupby, groupby))

            avg_trade_value = trade_value_sliding_window_4d / trade_count_sliding_window_4d # (timestamp // bar, 195, horizon_used // groupby, groupby)
            avg_trade_value_grouped = np.nansum(avg_trade_value, axis=-1) # (timestamp // bar, 195, horizon_used // groupby)

            return avg_trade_value_grouped

    def classify_B(B_df, classify_method_id):
        """Classify the B dataframe using specified method."""

        if classify_method_id == 0:
            """Dummy"""
            bar = 60
            full_timestamp = mask_future_data.index
            num_crypto = len(mask_future_data.columns)
            skipped_timestamp = full_timestamp[::bar]
            mask = np.ones((len(skipped_timestamp), num_crypto))

            return mask

        if classify_method_id == 2:
            """
            Classifies given 3D array based on cross-sectional mean, marking with either 1 or -1.
            :input: 3d array (timestamp, 195, horizon_used)
            :return: 3d array (timestamp, 195, horizon_used)
            """
            cross_section_mean = np.nanmean(B_df, axis=1) # (timestamp // bar, 24)
            cross_section_mean = cross_section_mean[:,np.newaxis,:] # (timestamp // bar, 1, 24)

            mask = ((B_df - cross_section_mean) > 0).astype(int) # (timestamp // bar, 195, 24)
            mask[mask == 0] = -1

            return mask

    def cal_factor(A_df, B_mask, calculation_method_id):
        """Calculate the factor based on A and B dataframes."""

        if calculation_method_id == 1:
            """
            Calculates the intra-group Pearson correlation coefficient based on given mask groups.

            # A: 3d array (time sequence, horizon_used, num_crypto)
            # mask: 2d array (time sequence, num_crypto)
            """

            horizon_used = 1440
            bar = 60

            # Calculates the intra-group Pearson correlation coefficient based on given mask groups.
            factor = np.empty((A_df.shape[0] - horizon_used // bar, A_df.shape[-1]))
            factor_df = pd.DataFrame(np.nan, index=mask_future_data.index, columns=mask_future_data.columns)
            
            full_timestamps = factor_df.index
            skipped_timestamps = full_timestamps[horizon_used:: bar]

            _incre = horizon_used // bar

            # Iterate over the time sequence excluding the tail part defined by "_incre"
            for t in tqdm(range(A_df.shape[0] - _incre)):
                classifications = B_mask[t +_incre]

                for class_id in np.unique(classifications):
                    crypto_indices = np.where(classifications == class_id)[0]
                    selected_cryptos_sequence = A_df[t + _incre, :, crypto_indices]
                    selected_cryptos_sequence = np.nan_to_num(selected_cryptos_sequence)
                    correlations = np.corrcoef(selected_cryptos_sequence, rowvar=True) # (num_crypto, num_crypto)
                    np.fill_diagonal(correlations, 0)
                    avg_correlation = np.nanmean(np.abs(correlations), axis=1) # (num_crypto,)
                    factor[t, crypto_indices] = avg_correlation
            
            factor_df.loc[skipped_timestamps] = factor
            
            return factor_df

    
    def finnna(data):
        """Fill NaN values in the provided data."""

        data = data.fillna(method='ffill', axis=0, limit=60)
        data = data.fillna(value=0)
        return data

    print(f'Total {len(idx_list)} items calculated here.')
    feature_dict = {}
    target_cache = {}  
    variable_cache = {}

    for i, idxx_list in enumerate(idx_list):

        print(f"Item {i+1}: Current indices using: {idxx_list}")

        if idxx_list[0] not in target_cache:
            start_time = time.time()
            target_cache[idxx_list[0]] = get_A(idxx_list[0])
            print(f"Time taken to find A: {time.time() - start_time}")
        Target_df = target_cache[idxx_list[0]]

        if idxx_list[1] not in variable_cache:
            start_time = time.time()
            variable_cache[idxx_list[1]] = get_B(idxx_list[1])
            print(f"Time taken to find B: {time.time() - start_time}")
        Variable_df = variable_cache[idxx_list[1]]

        start_time = time.time()
        Mask_df = classify_B(Variable_df, idxx_list[2])
        print(f"Time taken to find B_mask: {time.time() - start_time}")

        start_time = time.time()
        factor_df = cal_factor(Target_df, Mask_df, idxx_list[3])
        print(f"Time taken to Calculate factor: {time.time() - start_time}")

        feature_dict[f'factor_demo_l{int("".join(map(str, idxx_list)))}'] = finnna(factor_df)

    return feature_dict