import pandas as pd
from scipy.stats import pearsonr, ttest_ind, f_oneway

class HypothesisTester:
    def __init__(self, df_hyp: pd.DataFrame, config: dict):
        self.df_hyp = df_hyp.copy()
        self.config = config

    # -------------------- HYP 1 --------------------
    def run_h1_distance_vs_delay(self):
        """Hypothesis 1: Correlation between flight distance and arrival delay"""
        df = self.df_hyp.dropna(subset=['ARR_DELAY', 'DISTANCE'])
        corr, p = pearsonr(df['DISTANCE'], df['ARR_DELAY'])
        print("H1: Distance vs Arrival Delay")
        print(f"Pearson correlation coefficient: {corr:.3f}, p-value: {p:.3f}")
        if p < 0.05:
            if abs(corr) < 0.05:
                print("✅ Statistically significant but correlation is extremely weak; practically negligible.\n")
            else:
                print("✅ Significant correlation: flight distance is associated with delays.\n")
        else:
            print("❌ No significant correlation between distance and delay.\n")

    # -------------------- HYP 2 --------------------
    def run_h2_airline_ttest(self):
        """Hypothesis 2: Compare mean arrival delays between two airlines (t-test)"""
        airlines = self.config['hypothesis']['ttest_airlines']
        a = self.df_hyp.loc[self.df_hyp['AIRLINE'] == airlines[0], 'ARR_DELAY'].dropna()
        b = self.df_hyp.loc[self.df_hyp['AIRLINE'] == airlines[1], 'ARR_DELAY'].dropna()
        t_stat, p = ttest_ind(a, b)
        print(f"H2: {airlines[0]} vs {airlines[1]} mean arrival delays")
        print(f"T-statistic: {t_stat:.3f}, p-value: {p:.3f}")
        if p < 0.05:
            print(f"✅ Significant difference in mean delays between {airlines[0]} and {airlines[1]}.\n")
        else:
            print(f"❌ No significant difference in mean delays between {airlines[0]} and {airlines[1]}.\n")

    # -------------------- HYP 3 --------------------
    def run_h3_airline_anova(self):
        """Hypothesis 3: All airlines have the same mean arrival delay (ANOVA)"""
        groups = [self.df_hyp.loc[self.df_hyp['AIRLINE']==a, 'ARR_DELAY'].dropna()
                  for a in self.df_hyp['AIRLINE'].unique()]
        f_stat, p = f_oneway(*groups)
        print("H3: ANOVA across all airlines")
        print(f"F-statistic: {f_stat:.3f}, p-value: {p:.3f}")
        if p < 0.05:
            print("✅ Significant differences exist in delays between airlines.\n")
        else:
            print("❌ No significant differences in delays between airlines.\n")

    # -------------------- HYP 4 --------------------
    def run_h4_weather_vs_delay(self):
        """Hypothesis 4: Correlation between weather-related delays and arrival delays"""
        weather_col = self.config['hypothesis']['weather_col']
        df = self.df_hyp.dropna(subset=[weather_col, 'ARR_DELAY'])
        corr, p = pearsonr(df[weather_col], df['ARR_DELAY'])
        print(f"H4: {weather_col} vs Arrival Delay")
        print(f"Pearson correlation coefficient: {corr:.3f}, p-value: {p:.3f}")
        if p < 0.05:
            print("✅ Weather-related delays significantly impact arrival delays.\n")
        else:
            print("❌ Weather-related delays do not significantly impact arrival delays.\n")

    # -------------------- HYP 5 --------------------
    def run_h5_dep_hour_anova(self):
        """Hypothesis 5: Differences in mean arrival delays across scheduled departure hours"""
        df = self.df_hyp.copy()
        df['DEP_HOUR'] = df['CRS_DEP_TIME'] // 100
        hour_groups = [df.loc[df['DEP_HOUR'] == h, 'ARR_DELAY'].dropna()
                       for h in sorted(df['DEP_HOUR'].unique())]
        f_stat, p = f_oneway(*hour_groups)
        print("H5: Departure Hour vs Arrival Delay")
        print(f"F-statistic: {f_stat:.3f}, p-value: {p:.3f}")
        if p < 0.05:
            print("✅ Significant differences in mean arrival delays across departure hours.\n")
        else:
            print("❌ No significant differences in mean arrival delays across departure hours.\n")

    # -------------------- RUN ALL --------------------
    def run_all(self):
        print("\n" + "=" * 20 + " HYPOTHESIS TESTING " + "=" * 20)
        self.run_h1_distance_vs_delay()
        self.run_h2_airline_ttest()
        self.run_h3_airline_anova()
        self.run_h4_weather_vs_delay()
        self.run_h5_dep_hour_anova()