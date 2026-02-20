import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.stats import gmean

def test_make_results_table(check_normal=True):
    '''
    check_normal is introduced as a hack to be able to also check scib stuff..
    '''
    print("Starting test_make_results_table...")

    from scalpdemo import get_data, run_all, scalpscore
    # 1. Get data
    datasets, _ = get_data()
    print(f"Loaded {len(datasets)} datasets for testing.")

    # the later should be ts datasets... so we have a mix. and dont get nans :)
    if check_normal: datasets = datasets[:2] + datasets[-10:-8]
    else: datasets = datasets[18:21]

    # 2. Run all integration methods
    # Using a small subset of scalpvalues to speed up the test
    # And running on a single dataset for brevity
    # datasets_after_run, fnames, times = run_all([datasets[0], datasets[1]], scalpvalues=[.55, .75])
    datasets_after_run, fnames, times = run_all(datasets, scalpvalues=[.25, .55])

    # for ds in datasets_after_run:
    #     print(scalpscore.scib_scores(ds,'Scalp: 0.55'))
    # if not check_normal:
    #     return

    print(f"Finished running {len(fnames)} methods on {len(datasets_after_run)} datasets.")
    print("Methods run:", fnames)

    # 3. Calculate scalp scores
    scores = scalpscore(datasets_after_run)
    print("Calculated scalp scores.")
    return scores, datasets


    # print("Raw scores:", scores) # for debugging
    # 4. Define the chosen_scalp value for the test
    # This should match one of the 'Scalp: X.XX' strings in fnames
    chosen_scalp_option = '0.55' # Or '0.75' if you prefer
    # Filter fnames to get the exact Scalp method name
    full_scalp_method_name = [name for name in fnames if f'Scalp: {chosen_scalp_option}' in name][0]
    print(f"Chosen Scalp method for table: {full_scalp_method_name}")
    # 5. Call make_results_table
    results_df, latex_table = make_results_table(scores, datasets_after_run, chosen_scalp_option)
    print(latex_table)





def zehi_df(scores, df):
    '''
    Helper to convert scores dict to a DataFrame. If df is already a DataFrame, it returns it directly.
    '''
    if isinstance(df, pd.DataFrame):
        return df
    rows = []
    for ds, methods_dict in scores.items():
        for meth, vals in methods_dict.items():
            label = vals['label_mean']
            batch = vals['batch_mean']
            rows.append({'dataset': ds, 'method': meth, 'label': label, 'batch': batch})
    df =  pd.DataFrame(rows)
    # NaN check
    if df.isnull().any().any():
        print("Warning: NaN values found in scores DataFrame. This might affect calculations.")
        print(df_raw[df_raw.isnull().any(axis=1)])
    return df



def _calculate_subset_metrics(data_df, subset_ids, mode):
    if not subset_ids:
        return pd.DataFrame(columns=['label', 'batch', 'geomean']) # Empty df with expected columns

    subset_df = data_df[data_df['dataset'].isin(subset_ids)].copy()

    '''
    we produce a df. colmns are batch label and combined. the rows are the different methods.
    for rank, we use the ranks, and then comine by calculatingthe average rank over label+batch
    for geomean, we calculatethe geomean over the label scores, the geomean over the batch scores, and the geomean over both label and batch score columns
    '''

    if mode == 'rank':
        # Calculate ranks for 'label' and 'batch' scores
        ranks_label = subset_df.groupby('dataset')['label'].rank(ascending=False)
        ranks_batch = subset_df.groupby('dataset')['batch'].rank(ascending=False)

        # Combine label and batch ranks by averaging them per method and dataset
        subset_df['geomean'] = (ranks_label + ranks_batch) / 2
        subset_df['label'] = ranks_label
        subset_df['batch'] = ranks_batch

        # Average these ranks across datasets for each method
        return subset_df.groupby('method')[['label', 'batch', 'geomean']].mean()

    elif mode == 'geomean':

        # Calculate geometric mean of all 'label' scores for each method across *all datasets in the subset*
        label_geomeans = subset_df.groupby('method')['label'].apply(gmean).rename('label')
        # Calculate geometric mean of all 'batch' scores for each method across *all datasets in the subset*
        batch_geomeans = subset_df.groupby('method')['batch'].apply(gmean).rename('batch')

        def newmethod(x):
            return  gmean(np.concatenate([x['label'], x['batch']]))

        # Then, take the geometric mean of these combined_individual_geomean scores for each method
        combined_geomeans = subset_df.groupby('method').apply(newmethod).rename('geomean')

        # Combine these into a single DataFrame
        return pd.concat([label_geomeans, batch_geomeans, combined_geomeans], axis=1)
    else:
        raise ValueError("Mode must be 'rank' or 'geomean'")



def srt_methods(methods, chosen_scalp):
    """
    Sorts a list of method names. The chosen_scalp method comes first,
    followed by other Scalp methods (alphabetically), then other methods (alphabetically).

    Args:
        methods (list): A list of all method names (e.g., ['Harmony', 'Scalp: 0.55', 'Scanorama']).
        chosen_scalp (str): The specific Scalp method to prioritize (e.g., '0.55').

    Returns:
        list: A new list of sorted method names.
    """
    selected_scalp_full = [m for m in methods if f'Scalp: {chosen_scalp}' in m]
    other_scalp_methods = sorted([m for m in methods if 'Scalp' in m and m not in selected_scalp_full])
    non_scalp_methods = sorted([m for m in methods if 'Scalp' not in m])

    ordered_methods = []
    if selected_scalp_full:
        ordered_methods.extend(selected_scalp_full)
    ordered_methods.extend(other_scalp_methods)
    ordered_methods.extend(non_scalp_methods)

    return ordered_methods


def subselect_scalp_methods(chosen_scalp, all_scalp_methods):
    if not chosen_scalp:
        return all_scalp_methods
    chosen_scalp = chosen_scalp.split(' ')
    res = []
    for sel in chosen_scalp:
        for all_m in all_scalp_methods:
            if f'Scalp: {sel}' in all_m:
                res.append(all_m)
    return res

def make_results_table(scores, datasets, chosen_scalp,df=False, mode='rank', round_to =2 , dataselect = 'label batch total'):
    """
    scores: dict of dataset_id -> {method: {'label_mean':..., 'batch_mean':...}}
        -> alternatively there is the df argument
    datasets: list of AnnData or similar objects with .uns['timeseries']
    chosen_scalp: string, the exact scalp variant name to include
    mode: 'rank' or 'geomean'
    """

    #####################################
    #  SETUP
    ######################################
    df_raw = zehi_df(scores, df )

    # Identify timeseries and batch datasets
    dataset_ids = [d.uns.get('dataset_id', i) for i, d in enumerate(datasets)]
    ts_ids = [dataset_ids[i] for i, ds in enumerate(datasets) if ds.uns.get('timeseries', False)]
    batch_ids = [dataset_ids[i] for i, ds in enumerate(datasets) if not ds.uns.get('timeseries', False)]
    # print how many of which type we found
    print(f"Identified {len(ts_ids)} timeseries datasets and {len(batch_ids)} batch datasets. {len(dataset_ids)=}")


    # Filter methods: include all non-Scalp and the chosen Scalp variant
    all_methods = df_raw['method'].unique()
    non_scalp_methods = [m for m in all_methods if not m.startswith('Scalp')]
    all_scalp_methods = [m for m in all_methods if m.startswith('Scalp')]
    selected_scalp_methods = subselect_scalp_methods(chosen_scalp, all_scalp_methods)
    allowed_methods = non_scalp_methods + selected_scalp_methods
    df_filtered = df_raw[df_raw['method'].isin(allowed_methods)].copy()


    ############################################
    #
    #################################################


    # Calculate metrics for TS, Batch, and Total datasets
    ts_metrics = _calculate_subset_metrics(df_filtered, ts_ids, mode).add_prefix('TS_')
    batch_metrics = _calculate_subset_metrics(df_filtered, batch_ids, mode).add_prefix('Batch_')
    total_metrics = _calculate_subset_metrics(df_filtered, dataset_ids, mode).add_prefix('Total_')

    # Concatenate results
    results_df = pd.concat([ts_metrics, batch_metrics, total_metrics], axis=1)

    # Generate LaTeX table
    latex_table = format_res_T(results_df, selected_scalp_methods, round_to= round_to, show = dataselect, markminmax = min if mode == 'rank' else max)
    return results_df, latex_table


# rewrite the format_res_T function.

def format_res_T(results, selected_scalp_methods, round_to = 2, show= 'batch label total', markminmax = min):
    '''
    methods go to X axis.
    result columns are like this: Index(['TS_label_rank', 'TS_batch_rank', 'TS_geo_rank', 'Batch_label_rank',
       'Batch_batch_rank', 'Batch_geo_rank', 'Total_label_rank',
       'Total_batch_rank', 'Total_geo_rank'],
      dtype='object')

    index contains the methods

    we produce a latex table exactly like this: 4 columns , datapresented as slash separated tripplets a/b/c, numbers reduced to 1 decimal

    method | label | batch | total
    methodname1 | (total_label_rank/batch_label_rank/ ts_label_rank) | (same for batchh) | (same for geo)

    return  latex_code
    '''

    ######################
    # sort the table ...
    ############################

    methods = results.index.tolist()
    scalp_methods_in_results = [m for m in methods if 'Scalp' in m]
    other_methods = [m for m in methods if 'Scalp' not in m]
    ordered_methods = []
    ordered_methods.extend(sorted(selected_scalp_methods))
    ordered_methods.extend(sorted(other_methods))
    results = results.loc[ordered_methods].copy()




    # Create the new DataFrame structure
    formatted_data = defaultdict(list)
    show = show.split(' ')
    for method in ordered_methods:
        formatted_data['method'].append(method)


        # we need to change how the following work: 1. prepare strings for all  3 elements 2. only add show to the result, join with '/'
        #  do not be too smart.. just replace the next 3 blocks

        # Helper to get formatted string based on 'show' list
        def fmt_row(method, metric_name):
            vals = {
                'total': f"{results.loc[method, f'Total_{metric_name}']:.{round_to}f}",
                'batch': f"{results.loc[method, f'Batch_{metric_name}']:.{round_to}f}",
                'label': f"{results.loc[method, f'TS_{metric_name}']:.{round_to}f}" # TS is the label for timeseries datasets here
            }
            main_val = vals['total']
            sub_vals = "/".join([vals[s] for s in show if s in vals])
            return sub_vals

        formatted_data['label'].append(fmt_row(method, 'label'))
        formatted_data['batch'].append(fmt_row(method, 'batch'))
        formatted_data['total'].append(fmt_row(method, 'geomean'))



    df_formatted = pd.DataFrame(formatted_data)

    # Determine which entries to boldface (the minimum value for the 'Total' component of each column)
    latex_df = df_formatted.copy()

    for col_name in ['label', 'batch', 'total']:
        # contains one or more values seperated by /, we  find the maximum by column and make it textbf as below:

        # Extract the first part of the slash-separated string for comparison
        column_vals = latex_df[col_name].apply(lambda x: float(x.split('/')[0]))
        min_rank = markminmax(column_vals)

        new_col = []
        for val in latex_df[col_name]:
            parts = val.split('/')
            if np.isclose(float(parts[0]), min_rank):
                parts[0] = f"\\textbf{{{parts[0]}}}"
            new_col.append("/".join(parts))
        latex_df[col_name] = new_col

    latex_code = latex_df.to_latex(
        index=False,
        escape=False,
        column_format="lccc", # Adjust column format for visual separation
        header=[
            "Method", # Changed from "" to "Method"
            "Label",
            "Batch",
            "(g)Mean"
        ]
    )

    print(latex_code)
    return latex_code


