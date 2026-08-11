import pandas as pd
import scipy.stats as stats
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_prep_data(filepath):
    print("--- Loading and Prepping Data ---")
    df = pd.read_csv(filepath)

    # 1. Feature Engineering: Calculate Entity Length (Phonological Weight)
    df['entity_length'] = df['entity_text'].apply(lambda x: len(str(x).split()))

    # 2. Clean up roles: Keep only the core grammatical roles to reduce noise
    core_roles = ['nsubj', 'obj', 'obl', 'nmod', 'amod', 'root']
    df_core = df[df['dep_role'].isin(core_roles)].copy()

    print(f"Loaded {len(df)} total entities. Filtering to {len(df_core)} core syntactic roles.")
    return df_core


def test_syntax_hierarchy(df):
    print("\n=== Hypothesis 1: Syntactic Hierarchy (ANOVA) ===")
    # Does salience differ significantly based on the grammatical role?

    # Group data by role
    roles = df['dep_role'].unique()
    role_groups = [df[df['dep_role'] == role]['silver_salience'] for role in roles]

    # One-Way ANOVA
    f_stat, p_val = stats.f_oneway(*role_groups)
    print(f"ANOVA F-statistic: {f_stat:.4f}, p-value: {p_val:.4e}")

    if p_val < 0.05:
        print("Significant difference found! Running Tukey HSD post-hoc test...")
        tukey = pairwise_tukeyhsd(endog=df['silver_salience'], groups=df['dep_role'], alpha=0.05)
        print(tukey)
    else:
        print("No significant difference found across syntactic roles.")


def test_pos_differences(df):
    print("\n=== Hypothesis 2: Lexical vs. Named Entities (T-Test) ===")
    # Are Proper Nouns significantly more salient than Common Nouns?

    propn = df[df['pos_tag'] == 'PROPN']['silver_salience']
    noun = df[df['pos_tag'] == 'NOUN']['silver_salience']
    pron = df[df['pos_tag'] == 'PRON']['silver_salience']

    print(f"Mean Salience - Proper Nouns: {propn.mean():.4f} (n={len(propn)})")
    print(f"Mean Salience - Pronouns: {pron.mean():.4f} (n={len(pron)})")
    print(f"Mean Salience - Common Nouns: {noun.mean():.4f} (n={len(noun)})")

    # Independent T-Test
    t_stat, p_val = stats.ttest_ind(propn, noun, equal_var=False)  # Welch's t-test
    print(f"T-statistic comparing PROPN and NOUN: {t_stat:.4f}, p-value: {p_val:.4e}")

    t_stat, p_val = stats.ttest_ind(pron, noun, equal_var=False)  # Welch's t-test
    print(f"T-statistic comparing PRON and NOUN: {t_stat:.4f}, p-value: {p_val:.4e}")

    t_stat, p_val = stats.ttest_ind(propn, pron, equal_var=False)  # Welch's t-test
    print(f"T-statistic comparing PROPN and PRON: {t_stat:.4f}, p-value: {p_val:.4e}")

def test_entity_length(df):
    print("\n=== Hypothesis 3: Phonological Weight (Correlation) ===")
    # Do longer entities decay faster?

    # Using Spearman because length is ordinal/discrete and likely non-normal
    corr, p_val = stats.spearmanr(df['entity_length'], df['silver_salience'])
    print(f"Spearman Correlation: {corr:.4f}, p-value: {p_val:.4e}")


def run_mixed_effects_model(df):
    print("\n=== Mixed-Effects Regression ===")
    # Combines all features, controlling for Document ID

    # Formula: predict salience using length, POS, and Role.
    # C() indicates categorical variables.
    formula = "silver_salience ~ entity_length + C(pos_tag, Treatment(reference='NOUN')) + C(dep_role, Treatment(reference='nsubj'))"

    try:
        model = smf.mixedlm(formula, data=df, groups=df["doc_id"])
        result = model.fit()
        print(result.summary())
    except Exception as e:
        print(f"Mixed model failed (often happens if data is too small or singular): {e}")
        print("Falling back to standard Ordinary Least Squares (OLS) regression...")
        model = smf.ols(formula, data=df)
        result = model.fit()
        print(result.summary())


def plot_results(df):
    print("\n--- Generating Plots ---")
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Syntax
    sns.barplot(data=df, x='dep_role', y='silver_salience', ax=axes[0],
                order=['nsubj', 'obj', 'obl', 'nmod', 'amod', 'root'],
                palette='viridis', capsize=.1)
    axes[0].set_title('Salience by Syntactic Role')
    axes[0].set_ylabel('Probability of Survival (Silver Salience)')

    # Plot 2: POS
    sns.barplot(data=df[df['pos_tag'].isin(['NOUN', 'PROPN', 'PRON'])],
                x='pos_tag', y='silver_salience', ax=axes[1],
                palette='magma', capsize=.1)
    axes[1].set_title('Salience by Part of Speech')
    axes[1].set_ylabel('')

    # Plot 3: Length
    sns.regplot(data=df, x='entity_length', y='silver_salience', ax=axes[2],
                scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'}, x_jitter=0.2)
    axes[2].set_title('Salience vs. Entity Length')
    axes[2].set_ylabel('')

    plt.tight_layout()
    plt.savefig('salience_analysis_plots.png', dpi=300)
    print("Saved plots to 'salience_analysis_plots.png'")
    plt.show()


if __name__ == "__main__":
    csv_file = "gum_dev_salience_probing.csv"

    # Load data
    df = load_and_prep_data(csv_file)

    # Run statistical tests
    test_syntax_hierarchy(df)
    test_pos_differences(df)
    test_entity_length(df)
    run_mixed_effects_model(df)

    # Generate visual outputs
    plot_results(df)