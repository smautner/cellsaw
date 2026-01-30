

#import pandas as pd
#import numpy as np
#from collections import defaultdict
#from scalp import get_data, run_all, scalpscore
#from scipy.stats import gmean

#def test_make_results_table(check_normal=True):
#    '''
#    check_normal is introduced as a hack to be able to also check scib stuff..
#    '''
#    print("Starting test_make_results_table...")

#    # 1. Get data
#    datasets, _ = get_data()
#    print(f"Loaded {len(datasets)} datasets for testing.")

#    # the later should be ts datasets... so we have a mix. and dont get nans :)
#    if check_normal: datasets = datasets[:2] + datasets[-10:-8]
#    else: datasets = datasets[18:21]

#    # 2. Run all integration methods
#    # Using a small subset of scalpvalues to speed up the test
#    # And running on a single dataset for brevity
#    # datasets_after_run, fnames, times = run_all([datasets[0], datasets[1]], scalpvalues=[.55, .75])
#    datasets_after_run, fnames, times = run_all(datasets, scalpvalues=[.55, .75])


#    for ds in datasets_after_run:
#        print(scalp.score.scib_scores(ds,'Scalp: 0.55'))
#    if not check_normal:
#        return


#    print(f"Finished running {len(fnames)} methods on {len(datasets_after_run)} datasets.")
#    print("Methods run:", fnames)

#    # 3. Calculate scalp scores
#    scores = scalpscore(datasets_after_run)
#    print("Calculated scalp scores.")
#    # print("Raw scores:", scores) # for debugging

#    # 4. Define the chosen_scalp value for the test
#    # This should match one of the 'Scalp: X.XX' strings in fnames
#    chosen_scalp_option = '0.55' # Or '0.75' if you prefer

#    # Filter fnames to get the exact Scalp method name
#    full_scalp_method_name = [name for name in fnames if f'Scalp: {chosen_scalp_option}' in name][0]
#    print(f"Chosen Scalp method for table: {full_scalp_method_name}")


#    # 5. Call make_results_table
#    results_df, latex_table = make_results_table(scores, datasets_after_run, chosen_scalp_option)
#    print(latex_table)



#def make_results_table(scores, datasets, chosen_scalp, df=False):
#    """
#    scores: dict of dataset_id -> {method: {'label_mean':..., 'batch_mean':...}}
#    datasets: list of AnnData or similar objects with .uns['timeseries']
#    chosen_scalp: string, the exact scalp variant name to include
#    """
#    # -------- MAKE SURE WE HAVE A DF ----------
#    def  makeResultsTableDF(scores):
#        rows = []
#        for ds, methods_dict in scores.items():
#            for meth, vals in methods_dict.items():
#                label = vals['label_mean']
#                batch = vals['batch_mean']
#                #geo = gmean([label, batch])
#                # rows.append({'dataset': ds, 'method': meth, 'label': label, 'batch': batch, 'geomean': geo})
#                rows.append({'dataset': ds, 'method': meth, 'label': label, 'batch': batch})
#        return pd.DataFrame(rows)



#    if type(df) == bool :
#        df = makeResultsTableDF(scores)


#    # add geomean, this is a bit nonsense... lets see
#    df['geomean'] = df[['label', 'batch']].apply(gmean, axis=1)



#    # --------- NaN check,  WO IS TIMESERIES? ---------
#    full_df = df
#    if full_df.isnull().any().any():
#        print("Warning: NaN values found in scores DataFrame. This might affect ranking.")
#        print(full_df[full_df.isnull().any(axis=1)])

#    ts_ids = [int(i) for i, e in enumerate(datasets) if datasets[i].uns.get('timeseries', False)]
#    batch_ids = [int(i) for i, e in enumerate(datasets) if not datasets[i].uns.get('timeseries', False)]

#    # -------- FILTER METHODS ----------
#    allowed_methods = set(m for m in full_df['method'] if not m.startswith('Scalp'))
#    scalp_methods = [m for m in full_df['method'].unique() if m.startswith('Scalp')]
#    # If chosen_scalp is provided, filter for it
#    if chosen_scalp:
#        print(f"{ scalp_methods=}")
#        selected_scalp_methods = [m for m in scalp_methods if m.split()[-1] in   chosen_scalp.split()  ]
#    else:
#        selected_scalp_methods = scalp_methods
#    allowedscalp = set(selected_scalp_methods)
#    allowed_methods = allowed_methods.union(allowed_methods, allowedscalp)
#    df = full_df[full_df['method'].isin(allowed_methods)].copy()



#    # -------- Step 3: Compute ranks ----------
#    ranks = df.groupby('dataset')[['label', 'batch', 'geomean']].rank(ascending=False)
#    df[['label_rank', 'batch_rank', 'geo_rank']] = ranks

#    # check nans
#    if df.isnull().any().any():
#        print("Warning: NaN values found in ranks DataFrame. df is clean 1250")
#        print(ranks[ranks.isnull().any(axis=1)])

#    # -------- Step 4: Compute mean ranks ----------
#    def mean_ranks(ids):
#        sub = df[df['dataset'].isin(ids)]
#        return sub.groupby('method')[['label_rank', 'batch_rank', 'geo_rank']].mean()

#    ts_ranks = mean_ranks(ts_ids)
#    batch_ranks = mean_ranks(batch_ids)


#    total_ranks = df.groupby('method')[['label_rank', 'batch_rank', 'geo_rank']].mean()

#    result = pd.concat([
#        ts_ranks.add_prefix('TS_'),
#        batch_ranks.add_prefix('Batch_'),
#        total_ranks.add_prefix('Total_')
#    ], axis=1)


#    # check nan
#    if result.isnull().any().any():
#        print("Warning: NaN values found in ranks DataFrame. This might affect mean ranks.")
#        print(ranks[ranks.isnull().any(axis=1)])
#        breakpoint()

#    return result


#def format_res(results, selected_scalp_methods):
#    '''
#    here we had the methods in the y-axis..
#    '''

#    # -------- Step 5: Transpose ----------
#    result_t = results.T

#    # Reorder columns to place the chosen Scalp method first
#    cols = result_t.columns.tolist()
#    if selected_scalp_methods and selected_scalp_methods[0] in cols:
#        cols.remove(selected_scalp_methods[0])
#        cols.insert(0, selected_scalp_methods[0])
#        result_t = result_t[cols]

#    # -------- Step 6: Boldface best in each row ----------
#    latex_df = result_t.copy()
#    for row in latex_df.index:
#        vals = latex_df.loc[row]
#        minval = pd.to_numeric(vals, errors='coerce').min()
#        latex_df.loc[row] = [
#            f"\\textbf{{{float(v):.1f}}}" if np.isclose(float(v), minval) else f"{float(v):.1f}"
#            for v in vals
#        ]

#    latex_code = latex_df.to_latex(escape=False)
#    latex_code = latex_code.replace('_', ' ')

#    return result_t, latex_code




#def format_res_T(results, selected_scalp_methods):
#    '''
#    methods go to X axis.
#    result columns are like this: Index(['TS_label_rank', 'TS_batch_rank', 'TS_geo_rank', 'Batch_label_rank',
#       'Batch_batch_rank', 'Batch_geo_rank', 'Total_label_rank',
#       'Total_batch_rank', 'Total_geo_rank'],
#      dtype='object')

#    index contains the methods

#    we produce a latex table exactly like this: 4 columns , datapresented as slash separated tripplets a/b/c, numbers reduced to 1 decimal

#    method | label | batch | total
#    methodname1 | (total_label_rank/batch_label_rank/ ts_label_rank) | (same for batchh) | (same for geo)

#    return  latex_code
#    '''
#    # Create the new DataFrame structure
#    formatted_data = defaultdict(list)
#    methods = results.index.tolist()

#    methods.sort()
#    methods.insert(0, methods.pop()) # this works because of the alphabetical ordering...
#    print(f"{methods=}")
#    results = results.loc[methods].copy()

#    for method in methods:
#        formatted_data['method'].append(method)

#        # Collect ranks for 'label'
#        total_label = results.loc[method, 'Total_label_rank']
#        batch_label = results.loc[method, 'Batch_label_rank']
#        ts_label = results.loc[method, 'TS_label_rank']
#        #formatted_data['label'].append(f"{total_label:.1f}/{batch_label:.1f}/{ts_label:.1f}")
#        formatted_data['label'].append(f"{total_label:.1f}({batch_label:.1f}/{ts_label:.1f})")

#        # Collect ranks for 'batch'
#        total_batch = results.loc[method, 'Total_batch_rank']
#        batch_batch = results.loc[method, 'Batch_batch_rank']
#        ts_batch = results.loc[method, 'TS_batch_rank']
#        # formatted_data['batch'].append(f"{total_batch:.1f}/{batch_batch:.1f}/{ts_batch:.1f}")
#        formatted_data['batch'].append(f"{total_batch:.1f}({batch_batch:.1f}/{ts_batch:.1f})")

#        # Collect ranks for 'total' (geomean)
#        total_geo = results.loc[method, 'Total_geo_rank']
#        batch_geo = results.loc[method, 'Batch_geo_rank']
#        ts_geo = results.loc[method, 'TS_geo_rank']
#        # formatted_data['total'].append(f"{total_geo:.1f}/{batch_geo:.1f}/{ts_geo:.1f}")
#        formatted_data['total'].append(f"{total_geo:.1f}({batch_geo:.1f}/{ts_geo:.1f})")

#    df_formatted = pd.DataFrame(formatted_data)

#    # Determine which entries to boldface (the minimum value for each rank type within each column of the *original* results)
#    # This logic needs to consider the 'Total', 'Batch', 'TS' ranks separately for each 'label', 'batch', 'geo' group.
#    latex_df = df_formatted.copy()

#    for col_name in ['label', 'batch', 'total']:
#        # boldfacing the first value in the tipplets if it is lowest rank..


#        column_values = latex_df[col_name].apply(lambda x: float(x.split('(')[0]))  # Extract first value before '('
#        min_rank = column_values.min()

#        # Iterate through the rows to apply bolding based on the first value
#        # then only bold the first value itself and ignore the stuff in brackets

#        latex_df[col_name] = [
#            f"\\textbf{{{val[:3]}}}{val[3:]}" if np.isclose(float(val.split('(')[0]), min_rank) else val for val in latex_df[col_name]
#        ]


#    def rename_method(method_name):
#        if 'Scalp' in method_name:
#            return method_name[-4:] # e.g., '0.75'
#        else:
#            return method_name[:4]   # e.g., 'H' for Harmony, 'S' for Scanorama

#    latex_df['method'] = latex_df['method'].apply(rename_method)




#    latex_code = latex_df.to_latex(
#        index=False,
#        escape=False,
#        column_format="lccc", # Adjust column format for visual separation
#        header=[
#            "",
#            "Label",
#            "Batch",
#            "Mean"
#        ]
#    )


#    return latex_code



import pandas as pd
import numpy as np
from collections import defaultdict
from scalp import get_data, run_all, scalpscore
from scipy.stats import gmean

def test_make_results_table(check_normal=True):
    '''
    check_normal is introduced as a hack to be able to also check scib stuff..
    '''
    print("Starting test_make_results_table...")

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
    datasets_after_run, fnames, times = run_all(datasets, scalpvalues=[.55, .75])


    for ds in datasets_after_run:
        print(scalpscore.scib_scores(ds,'Scalp: 0.55'))
    if not check_normal:
        return


    print(f"Finished running {len(fnames)} methods on {len(datasets_after_run)} datasets.")
    print("Methods run:", fnames)

    # 3. Calculate scalp scores
    scores = scalpscore(datasets_after_run)
    print("Calculated scalp scores.")
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



def make_results_table(scores, datasets, chosen_scalp,df=False, mode='rank'):
    """
    scores: dict of dataset_id -> {method: {'label_mean':..., 'batch_mean':...}}
    datasets: list of AnnData or similar objects with .uns['timeseries']
    chosen_scalp: string, the exact scalp variant name to include
    mode: 'rank' or 'geomean'
    """

    # Helper to convert scores dict to a DataFrame
    def _scores_to_df(scores_dict):
        rows = []
        for ds, methods_dict in scores_dict.items():
            for meth, vals in methods_dict.items():
                label = vals['label_mean']
                batch = vals['batch_mean']
                rows.append({'dataset': ds, 'method': meth, 'label': label, 'batch': batch})
        return pd.DataFrame(rows)

    if type(df) == bool :
        df_raw = _scores_to_df(scores)
    else:
        df_raw = df

    # NaN check
    if df_raw.isnull().any().any():
        print("Warning: NaN values found in scores DataFrame. This might affect calculations.")
        print(df_raw[df_raw.isnull().any(axis=1)])

    # Identify timeseries and batch datasets
    dataset_ids = [d.uns.get('dataset_id', i) for i, d in enumerate(datasets)]
    ts_ids = [dataset_ids[i] for i, ds in enumerate(datasets) if ds.uns.get('timeseries', False)]
    batch_ids = [dataset_ids[i] for i, ds in enumerate(datasets) if not ds.uns.get('timeseries', False)]

    # Filter methods: include all non-Scalp and the chosen Scalp variant
    all_methods = df_raw['method'].unique()
    non_scalp_methods = [m for m in all_methods if not m.startswith('Scalp')]
    selected_scalp_methods = [m for m in all_methods if f'Scalp: {chosen_scalp}' in m]

    if not selected_scalp_methods:
        print(f"Warning: Chosen Scalp option '{chosen_scalp}' not found in method list. Including all Scalp methods.")
        allowed_methods = non_scalp_methods + [m for m in all_methods if m.startswith('Scalp')]
    else:
        allowed_methods = non_scalp_methods + selected_scalp_methods

    df_filtered = df_raw[df_raw['method'].isin(allowed_methods)].copy()

    # Sub-function to calculate averages for dataset subsets
    def _calculate_subset_metrics(data_df, subset_ids, mode):
        if not subset_ids:
            return pd.DataFrame(columns=['label', 'batch', 'geomean']) # Empty df with expected columns

        subset_df = data_df[data_df['dataset'].isin(subset_ids)].copy()

        if mode == 'rank':
            # Rank within the subset
            ranks = subset_df.groupby('dataset')[['label', 'batch']].rank(ascending=False)
            subset_df[['label_rank', 'batch_rank']] = ranks

            # Compute geomean of ranks
            subset_df['geomean_rank'] = subset_df[['label_rank', 'batch_rank']].apply(gmean, axis=1)

            # Average ranks across datasets in the subset
            return subset_df.groupby('method')[['label_rank', 'batch_rank', 'geomean_rank']].mean()

        elif mode == 'geomean':
            # Directly compute geomean of scores (label/batch)
            subset_df['geomean_score'] = subset_df[['label', 'batch']].apply(gmean, axis=1)

            # Average scores across datasets in the subset
            return subset_df.groupby('method')[['label', 'batch', 'geomean_score']].mean()
        else:
            raise ValueError("Mode must be 'rank' or 'geomean'")

    # Calculate metrics for TS, Batch, and Total datasets
    ts_metrics = _calculate_subset_metrics(df_filtered, ts_ids, mode).add_prefix('TS_')
    batch_metrics = _calculate_subset_metrics(df_filtered, batch_ids, mode).add_prefix('Batch_')

    # Calculate total metrics
    total_metrics = _calculate_subset_metrics(df_filtered, dataset_ids, mode).add_prefix('Total_')

    # Concatenate results
    results_df = pd.concat([ts_metrics, batch_metrics, total_metrics], axis=1)

    # Rename columns based on mode for consistency
    if mode == 'rank':
        results_df = results_df.rename(columns={
            c: c.replace('_rank', '') for c in results_df.columns if '_rank' in c
        })
    elif mode == 'geomean':
        results_df = results_df.rename(columns={
            c: c.replace('_score', '') for c in results_df.columns if '_score' in c
        })

    # Generate LaTeX table
    latex_table = format_res_T(results_df, selected_scalp_methods)

    return results_df, latex_table




def format_res_T(results, selected_scalp_methods):
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
    # Create the new DataFrame structure
    formatted_data = defaultdict(list)
    methods = results.index.tolist()

    # Ensure the selected_scalp_methods appear first, then sort alphabetically
    # Assuming selected_scalp_methods contains full method names like 'Scalp: 0.55'

    # Separate Scalp and non-Scalp methods for ordering
    scalp_methods_in_results = [m for m in methods if 'Scalp' in m]
    other_methods = [m for m in methods if 'Scalp' not in m]

    # Order: selected scalp first, then other scalp, then others alphabetically
    ordered_methods = []
    if selected_scalp_methods:
        # Add the specific chosen scalp method first
        for s_m in selected_scalp_methods:
            if s_m in methods:
                ordered_methods.append(s_m)

        # Remove selected scalp from other scalp methods to avoid duplication
        scalp_methods_in_results = [m for m in scalp_methods_in_results if m not in ordered_methods]

    # Add remaining scalp methods (alphabetically)
    ordered_methods.extend(sorted(scalp_methods_in_results))

    # Add other methods (alphabetically)
    ordered_methods.extend(sorted(other_methods))

    # Remove duplicates while preserving order (if any)
    ordered_methods = list(dict.fromkeys(ordered_methods))

    results = results.loc[ordered_methods].copy()


    for method in ordered_methods:
        formatted_data['method'].append(method)

        # Collect ranks for 'label'
        total_label = results.loc[method, 'Total_label']
        batch_label = results.loc[method, 'Batch_label']
        ts_label = results.loc[method, 'TS_label']
        formatted_data['label'].append(f"{total_label:.1f}({batch_label:.1f}/{ts_label:.1f})")

        # Collect ranks for 'batch'
        total_batch = results.loc[method, 'Total_batch']
        batch_batch = results.loc[method, 'Batch_batch']
        ts_batch = results.loc[method, 'TS_batch']
        formatted_data['batch'].append(f"{total_batch:.1f}({batch_batch:.1f}/{ts_batch:.1f})")

        # Collect ranks for 'total' (geomean)
        total_geo = results.loc[method, 'Total_geomean']
        batch_geo = results.loc[method, 'Batch_geomean']
        ts_geo = results.loc[method, 'TS_geomean']
        formatted_data['total'].append(f"{total_geo:.1f}({batch_geo:.1f}/{ts_geo:.1f})")

    df_formatted = pd.DataFrame(formatted_data)

    # Determine which entries to boldface (the minimum value for the 'Total' component of each column)
    latex_df = df_formatted.copy()

    for col_name in ['label', 'batch', 'total']:
        # Extract the 'Total' value (first part before '(') for comparison
        column_total_values = latex_df[col_name].apply(lambda x: float(x.split('(')[0]))
        min_total_rank = column_total_values.min()

        latex_df[col_name] = [
            # Only bold the total component, leave the bracketed values as they are
            f"\\textbf{{{val.split('(')[0]}}}({val.split('(')[1]}" if np.isclose(float(val.split('(')[0]), min_total_rank) else val
            for val in latex_df[col_name]
        ]

    def rename_method(method_name):
        if 'Scalp' in method_name:
            return method_name.split(': ')[1] # e.g., '0.75' from 'Scalp: 0.75'
        else:
            return method_name # Keep other method names as is

    latex_df['method'] = latex_df['method'].apply(rename_method)

    latex_code = latex_df.to_latex(
        index=False,
        escape=False,
        column_format="lccc", # Adjust column format for visual separation
        header=[
            "Method", # Changed from "" to "Method"
            "Label",
            "Batch",
            "Mean"
        ]
    )

    return latex_code

```
