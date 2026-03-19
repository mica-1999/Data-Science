import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import os

warnings.simplefilter(action='ignore', category=FutureWarning)

class EDAAnalyzer:
    def __init__(self, df: pd.DataFrame, config: dict):
        self.df = df.copy()
        self.config = config
        self.numeric_cols_summary = config['eda']['numeric_summary']
        self.numeric_cols_hist = config['eda']['numeric_hist']
        self.scatter_plots = config['eda']['scatter_plots']
        self.output_dir = config['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)

        self.post_fe_hist_cols = config['eda_post_engineering']['numeric_hist']
        self.post_fe_corr_cols = config['eda_post_engineering']['correlation_cols']
        self.post_fe_boxplot_cols = config['eda_post_engineering']['boxplot']
        self.output_dir_post_fe = config['output_dir_post_fe']
        os.makedirs(self.output_dir_post_fe, exist_ok=True)

    # -------------------- BASIC SUMMARY STATS --------------------
    def summary_statistics(self):
        """Print basic stats, median, mean, std for numeric columns."""
        print("\nBasic statistics:\n", self.df[self.numeric_cols_summary].describe())

    # -------------------- TOP AIRLINES --------------------
    def plot_top_airlines(self, top_n=10):
        """Count plot for top N airlines."""
        plt.figure(figsize=(12,6))
        sns.countplot(
            y='AIRLINE',
            data=self.df,
            order=self.df['AIRLINE'].value_counts().index[:top_n]
        )
        plt.title(f"Top {top_n} Airlines by Number of Flights")
        plt.xlabel("Count")
        plt.ylabel("Airline")
        plt.savefig(os.path.join(self.output_dir, "top_airlines_count.png"), bbox_inches='tight')
        plt.close()

    # -------------------- HISTOGRAM PLOTS --------------------
    def plot_histograms(self):
        """Plot histograms for numeric columns."""
        for col in self.numeric_cols_hist:
            plt.figure(figsize=(8,4))
            sns.histplot(self.df[col], bins=50, kde=True)
            plt.title(f"Distribution of {col}")
            plt.savefig(os.path.join(self.output_dir, f"hist_{col}.png"), bbox_inches='tight')
            plt.close()

    # -------------------- BOXPLOT --------------------
    def plot_boxplots_by_airline(self, target_cols=['ARR_DELAY']):
        """Boxplots of target columns by airline."""
        for col in target_cols:
            plt.figure(figsize=(12,6))
            sns.boxplot(x='AIRLINE', y=col, data=self.df)
            plt.ylim(-100, 100)
            plt.xticks(rotation=45)
            plt.savefig(os.path.join(self.output_dir, f"boxplot_{col}_by_airline.png"), bbox_inches='tight')
            plt.close()

    # -------------------- FEATURES CORRELATION --------------------
    def plot_correlation_heatmap(self):
        """Correlation matrix & heatmap for numeric columns."""
        corr_matrix = self.df[self.numeric_cols_hist].corr()
        plt.figure(figsize=(12,4))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.title("Correlation Heatmap")
        plt.savefig(os.path.join(self.output_dir, "corr_heatmap.png"), bbox_inches='tight')
        plt.close()

    # -------------------- KDE PLOT --------------------
    def plot_kde_by_airline(self, target_cols=['ARR_DELAY']):
        """KDE plots of target columns per airline."""
        for col in target_cols:
            plt.figure(figsize=(12,6))
            sns.kdeplot(data=self.df, x=col, hue='AIRLINE', fill=True, alpha=0.5)
            plt.title(f"KDE Plot of {col} by Airline")
            plt.xlabel(col)
            plt.ylabel('Density')
            plt.savefig(os.path.join(self.output_dir, f"kde_{col}_by_airline.png"), bbox_inches='tight')
            plt.close()

    # -------------------- SCATTER PLOTS --------------------
    def plot_scatter(self, scatter_configs=None):
        """Generate scatter plots as per config list."""
        if scatter_configs is None:
            scatter_configs = self.scatter_plots

        for x_col, y_col, title, filename in scatter_configs:
            plt.figure(figsize=(8,6))
            sns.scatterplot(x=x_col, y=y_col, data=self.df, alpha=0.3)
            plt.title(f"Scatter Plot: {title}")
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.savefig(os.path.join(self.output_dir, filename), bbox_inches='tight')
            plt.close()

    # -------------------- POST FEATURE ENGINEERING EDA --------------------
    def plot_post_fe_histograms(self):
        """Plot histograms for engineered feature columns."""
        for col in self.post_fe_hist_cols:
            if col not in self.df.columns:
                print(f"⚠️  Column '{col}' not found in dataset, skipping...")
                continue
            plt.figure(figsize=(8, 4))
            sns.histplot(self.df[col], bins=30, kde=True)
            plt.title(f"Distribution of {col} (Engineered)")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.savefig(os.path.join(self.output_dir_post_fe, f"hist_eng_{col}.png"), bbox_inches='tight')
            plt.close()

    def plot_post_fe_correlation_heatmap(self):
        """Correlation heatmap for engineered features vs ARR_DELAY."""
        valid_cols = [c for c in self.post_fe_corr_cols if c in self.df.columns]
        if not valid_cols:
            print("⚠️  No valid columns found for post-FE correlation heatmap, skipping...")
            return
        corr_matrix = self.df[valid_cols].corr()
        plt.figure(figsize=(12, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.title("Correlation Heatmap - Engineered Features")
        plt.savefig(os.path.join(self.output_dir_post_fe, "corr_heatmap_engineered.png"), bbox_inches='tight')
        plt.close()

    def plot_post_fe_boxplots(self):
        """Boxplots of ARR_DELAY grouped by key engineered binary/categorical features."""
        for col in self.post_fe_boxplot_cols:
            if col not in self.df.columns or 'ARR_DELAY' not in self.df.columns:
                print(f"⚠️  Column '{col}' or 'ARR_DELAY' not found, skipping boxplot...")
                continue
            plt.figure(figsize=(8, 5))
            sns.boxplot(x=col, y='ARR_DELAY', data=self.df)
            plt.title(f"ARR_DELAY by {col}")
            plt.xlabel(col)
            plt.ylabel("Arrival Delay (min)")
            plt.savefig(os.path.join(self.output_dir_post_fe, f"boxplot_arrdelay_by_{col}.png"), bbox_inches='tight')
            plt.close()

    def run_post_engineering(self):
        """Run EDA on engineered features after feature engineering step."""
        print("\n" + "=" * 20 + " POST-ENGINEERING EDA " + "=" * 20)

        print("\n1️⃣  Plotting histograms for engineered features...")
        self.plot_post_fe_histograms()

        print("2️⃣  Plotting correlation heatmap for engineered features...")
        self.plot_post_fe_correlation_heatmap()

        print("3️⃣  Plotting ARR_DELAY boxplots by engineered categorical features...")
        self.plot_post_fe_boxplots()

        print("🎉 Post-engineering EDA complete. All outputs saved to:", self.output_dir_post_fe)

    # -------------------- RUN ALL --------------------
    def run_all(self):
        """Run all EDA plots and stats with clear prints for each step."""
        print("\n" + "=" * 20 + " EDA STEP " + "=" * 20)

        print("\n1️⃣  Computing summary statistics...")
        self.summary_statistics()

        print("2️⃣  Plotting top airlines by flight count...")
        self.plot_top_airlines()

        print("3️⃣  Plotting histograms for numeric columns...")
        self.plot_histograms()

        print("4️⃣  Plotting boxplots of ARR_DELAY by airline...")
        self.plot_boxplots_by_airline()

        print("5️⃣  Plotting correlation heatmap for numeric features...")
        self.plot_correlation_heatmap()

        print("6️⃣  Plotting KDE of ARR_DELAY by airline...")
        self.plot_kde_by_airline()

        print("7️⃣  Generating scatter plots as per config...")
        self.plot_scatter()

        print("🎉 EDA complete. All outputs saved to:", self.output_dir)