import tweet_clustering as tc
import pandas as pd

if __name__ == '__main__':
    google_sheet_id = "1vp2DSvqIo36CJ3xR2ukm5PD3AFExu3uFyi3WNkoTshI"
    sheet_name = "Data"
    google_sheet_url = "https://docs.google.com/spreadsheets/d/{}/gviz/tq?tqx=out:csv&sheet={}".format(google_sheet_id,
                                                                                                       sheet_name)
    df = pd.read_csv(google_sheet_url, header=None)
    k_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for k in k_values:
        algo = tc.KMeans()
        algo.fit(df, k=k, random_seed=123)
        # algo.print_clusters()  # Uncomment if you want to view clusters
        # print()
        algo.print_cluster_count()
        print()
        print('SSE: ' + str(algo.sse()))
