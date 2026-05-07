import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import stanza
import pandas as pd
import os
import glob

class DiscourseSimulator:
    def __init__(self, model_name="meta-llama/Llama-3.2-1B"):
        """
        Initializes the LLM generator and the Stanza NLP pipeline.
        """
        print("Loading LLM for text generation...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.device = torch.device("cuda")

        print("Loading Stanza pipeline (including Coref and Depparse)...")
        # Ensure 'coref' is in the processors list
        # 'mwt' (Multi-Word Token) is required for some languages, recommended for English UD
        self.nlp = stanza.Pipeline(
            lang='en',
            processors='tokenize,mwt,pos,lemma,depparse,ner,coref',
            tokenize_pretokenized=True,
            package={"coref": "gum-nospeakers_roberta-large-lora"},
            config={'coref': {"plateau_epochs": 50}}
        )

    def generate_continuations(self, context, num_samples=10, max_new_tokens=50):
        """
        Generates diverse next-sentence continuations.
        """
        inputs = self.tokenizer(context, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            num_return_sequences=num_samples,
            pad_token_id=self.tokenizer.eos_token_id
        )

        continuations = []
        context_length = inputs['input_ids'].shape[1]

        for output in outputs:
            text = self.tokenizer.decode(output[context_length:], skip_special_tokens=True)
            # Take the first sentence only
            first_sentence = text.split('. ')[0] + '.' if '. ' in text else text
            continuations.append(first_sentence.strip())

        return continuations


    def analyze_salience(self, context, continuations, target_entity_text):
        """
        Uses Stanza to track entity clusters across context and generated continuations.
        """
        mentions_count = 0
        context_flat_string = " ".join([" ".join(sent) for sent in context])
        context_char_len = len(context_flat_string)
        all_competitor_data = []
        target_entity_lower = target_entity_text.lower()

        for continuation in continuations:
            if not continuation:
                continue

            # Optimized list comprehension: removed redundant (sent + "") string concat
            full_text = context + [sent.split(" ") for sent in continuation.split(". ")]
            doc = self.nlp(full_text)

            target_cluster_id = None
            target_chain = None

            # 1. Identify the cluster ID for our target entity in the context
            for chain in doc.coref:
                for mention in chain.mentions:
                    sent_idx = getattr(mention, 'sent_index', getattr(mention, 'sentence', 0))
                    sentence = doc.sentences[sent_idx]
                    mention_words = sentence.words[mention.start_word: mention.end_word]

                    if not mention_words:
                        continue

                    # Fast-fail: skip string joining and lowering if mention is outside context
                    if mention_words[0].start_char >= context_char_len:
                        continue

                    mention_text = " ".join([w.text for w in mention_words])

                    if target_entity_lower in mention_text.lower():
                        target_chain = chain
                        break  # Found our target mention!

                if target_chain:
                    break  # Break the outer loop since we found the target chain

            survived = False
            current_continuation_competitors = 0

            # 2. Check for survival and identify competitors
            for chain in doc.coref:
                is_target = (chain == target_chain)

                for mention in chain.mentions:
                    sent_idx = getattr(mention, 'sent_index', getattr(mention, 'sentence', 0))
                    sentence = doc.sentences[sent_idx]
                    mention_words = sentence.words[mention.start_word: mention.end_word]

                    # Fast-fail: completely skip empty mentions or context mentions here
                    if not mention_words or mention_words[0].start_char < context_char_len:
                        continue

                    if is_target:
                        if not survived:
                            mentions_count += 1
                            survived = True
                    else:
                        head_word = sentence.words[mention.start_word]
                        all_competitor_data.append({
                            "text": " ".join([w.text for w in mention_words]),
                            "role": head_word.deprel,
                            "pos": head_word.upos,
                            "is_pronoun": head_word.upos == 'PRON'
                        })
                        current_continuation_competitors += 1

        # Safe division fallback prevents ZeroDivisionError if continuations is empty/filtered
        salience_prob = mentions_count / len(continuations) if continuations else 0.0

        return {
            "salience_probability": salience_prob,
            "competitor_mentions": all_competitor_data,
            "avg_competitor_pressure": len(all_competitor_data) / len(continuations) if continuations else 0.0
        }


def test():
    sim = DiscourseSimulator()
    context = [["John", "Bauer", "works", "at", "Stanford", ".",], ["He", "has", "been", "there", "four", "years","."]]

    context_flat = ' '.join([' '.join(sent).replace(" .", ".") for sent in context])

    conts = sim.generate_continuations(context_flat, num_samples=5)
    target = "John Bauer"
    results = sim.analyze_salience(context, conts, target)

    print(f"\nSalience Score for {target}: {results['salience_probability']}")
    print(f"Competitor Pressure for {target}: {results['avg_competitor_pressure']}")

    target = "Stanford"
    results = sim.analyze_salience(context, conts, target)

    print(f"\nSalience Score for {target}: {results['salience_probability']}")
    print(f"Competitor Pressure for {target}: {results['avg_competitor_pressure']}")


def load_gum_tsv(filepath):
    """
    Parses a GUM coref TSV file and returns a list of pretokenized sentences.
    Format: [["Word", "1"], ["Word", "2", "."]]
    """
    sentences = []
    current_sentence = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # The #Text= header indicates the start of a new sentence
            if line.startswith('#Text='):
                # If we were building a sentence, save it before starting the new one
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                continue

            # Skip any other metadata comments
            if line.startswith('#'):
                continue

            # Split by tabs
            columns = line.split('\t')

            # In this TSV, Column 3 (Index 2) is the actual token string
            if len(columns) > 2:
                word = columns[2]
                current_sentence.append(word)

    # Catch the very last sentence in the file
    if current_sentence:
        sentences.append(current_sentence)

    return sentences


def run_document_pilot(tsv_filepath, target_entity_text, window_size=3):
    """
    Runs the sliding window discourse decay simulation over a full document.
    """
    print(f"Loading GUM document: {tsv_filepath}")
    doc_sentences = load_gum_tsv(tsv_filepath)
    print(f"Total sentences found: {len(doc_sentences)}")

    # Initialize your updated simulator
    sim = DiscourseSimulator()

    # This list will become your Salience Matrix
    salience_matrix = []

    # Start loop at `window_size` so we always have a full context block
    for i in range(window_size, len(doc_sentences)):
        print(f"\n--- Processing Window for Next Sentence {i + 1} ---")

        # 1. Grab the preceding k sentences
        context_window = doc_sentences[i - window_size: i]

        # Flatten context for the LLM Generator prompt
        # (A simple heuristic to fix punctuation spacing for the LLM)
        context_flat = ' '.join([' '.join(sent) for sent in context_window])
        context_flat = context_flat.replace(" .", ".").replace(" ,", ",")

        # 2. Generate Alternative Continuations (Silver Data)
        print("  Generating alternatives...")
        conts = sim.generate_continuations(context_flat, num_samples=10)  # 10 for paper quality

        # 3. Analyze Salience & Competitor Pressure
        print("  Analyzing discourse decay...")

        results = sim.analyze_salience(context_window, conts, target_entity_text)

        # 4. Log the data row
        salience_matrix.append({
            "sentence_idx": i,  # The index of the *next* sentence being predicted
            "context": context_flat,
            "target_entity": target_entity_text,
            "salience_probability": results["salience_probability"],
            "competitor_pressure": results["avg_competitor_pressure"],
            "continuations": '<>'.join(conts)
        })
        print(
            f"  Salience: {results['salience_probability']:.2f} | Pressure: {results['avg_competitor_pressure']:.2f}")



    # Convert to a DataFrame for easy saving/analysis
    df = pd.DataFrame(salience_matrix)
    return df


def single_file_pilot():
    # Replace with your actual GUM file path
    file_path = "GUM_news_korea.tsv"

    # For this pilot, pick an entity you know is heavily discussed in the document
    target = "North Korea"

    results_df = run_document_pilot(file_path, target, window_size=3)

    # Save the output to a CSV so you can graph the decay curve!
    results_df.to_csv("salience_matrix_pilot.csv", index=False)
    print("\nSimulation complete. Data saved to salience_matrix_pilot.csv")
    print(results_df.head(10))


def get_span_head(mention, sentence):
    """
    Finds the syntactic head of a word span.
    The head is the word whose dependency parent is outside the span.
    """
    start_word_idx = mention.start_word
    end_word_idx = mention.end_word

    span_words = sentence.all_words[start_word_idx:end_word_idx]

    # Stanza's word.head is a 1-based index, so we convert our 0-based span bounds
    span_start_1based = start_word_idx + 1
    span_end_1based = end_word_idx

    for word in span_words:
        # If the word's parent index is outside our span, this word is the root of the phrase!
        if word.head < span_start_1based or word.head > span_end_1based:
            return word

    # Fallback for edge-case parse errors: return the right-most word (usually the noun in English)
    return span_words[-1] if span_words else None


def analyze_window_entities(sim, context, continuations):
    """
    Auto-discovers entities in the final sentence of the context and
    calculates their survival rate in the continuations.
    """
    # 1. Parse the context to find the "launchpad" entities
    context_flat = " ".join([" ".join(sent) for sent in context])
    # Reconstruct list-of-lists for Stanza
    full_text = context + [sent.split(" ") for cont in continuations if cont for sent in cont.split(". ")]
    doc = sim.nlp(full_text)

    launchpad_sent_idx = len(context) - 1  # The last sentence of the context
    context_char_len = len(context_flat)

    active_entities = []

    # 2. Extract Independent Variables (Features from the context)
    for chain in doc.coref:
        last_mention_in_context = None

        # Look for the latest mention of this entity in the launchpad sentence
        for mention in chain.mentions:
            sent_idx = getattr(mention, 'sent_index', getattr(mention, 'sentence', 0))
            if sent_idx == launchpad_sent_idx:
                last_mention_in_context = mention

        if last_mention_in_context:
            sentence = doc.sentences[launchpad_sent_idx]
            mention_words = sentence.words[last_mention_in_context.start_word: last_mention_in_context.end_word]

            if not mention_words:
                continue

            head_word = get_span_head(last_mention_in_context, sentence)

            active_entities.append({
                "target_chain": chain,
                "text": " ".join([w.text for w in mention_words]),
                "role": head_word.deprel,  # Syntactic role (nsubj, obj, obl)
                "pos": head_word.upos,  # Part of speech (PRON, PROPN, NOUN)
                "is_pronoun": head_word.upos == 'PRON',
                "mentions_count": 0  # Initialize survival counter
            })

    # 3. Extract Dependent Variable (Survival in Continuations)
    valid_continuations = [c for c in continuations if c]

    for entity in active_entities:
        survived_continuations = 0
        target_chain = entity["target_chain"]

        for mention in target_chain.mentions:
            sent_idx = getattr(mention, 'sent_index', getattr(mention, 'sentence', 0))
            sentence = doc.sentences[sent_idx]
            mention_words = sentence.words[mention.start_word: mention.end_word]

            if not mention_words:
                continue

            # If the mention occurs in the LLM-generated portion
            if mention_words[0].start_char >= context_char_len:
                # We just need to know if it survived at least once in this continuation
                survived_continuations += 1
                # Note: Stanza groups all generated text into doc.sentences.
                # A more precise script would slice doc by continuation, but
                # grouping them allows Stanza's coref to resolve across all silver text simultaneously.

        # Calculate silver salience
        entity["silver_salience"] = survived_continuations / len(valid_continuations) if valid_continuations else 0.0

    return active_entities


def batch_process_dev_set(dev_directory, window_size=3):
    """
    Loops through all GUM TSV files in a directory and builds the Salience Matrix.
    """
    sim = DiscourseSimulator()
    salience_matrix = []

    # Find all TSV files in the dev folder
    tsv_files = glob.glob(os.path.join(dev_directory, "*.tsv"))
    print(f"Found {len(tsv_files)} files in {dev_directory}")

    for filepath in tsv_files:
        doc_id = os.path.basename(filepath).replace(".tsv", "")
        print(f"\nProcessing Document: {doc_id}")

        doc_sentences = load_gum_tsv(filepath)

        if len(doc_sentences) <= window_size:
            print(f"  Skipping {doc_id}: Not enough sentences.")
            continue

        for i in range(window_size, len(doc_sentences)):
            context_window = doc_sentences[i - window_size: i]
            context_flat = ' '.join([' '.join(sent) for sent in context_window])

            # Generate Silver Data
            conts = sim.generate_continuations(context_flat, num_samples=10)

            # Auto-extract features and salience
            active_entities = analyze_window_entities(sim, context_window, conts)

            # Append to our master dataset
            for entity in active_entities:
                salience_matrix.append({
                    "doc_id": doc_id,
                    "context_start_idx": i - window_size,
                    "launchpad_sent_idx": i - 1,
                    "entity_text": entity["text"],
                    "dep_role": entity["role"],
                    "pos_tag": entity["pos"],
                    "is_pronoun": entity["is_pronoun"],
                    "silver_salience": entity["silver_salience"]
                })
                print(
                    f"  [{i}] Tracked '{entity['text']}' ({entity['role']}) -> Salience: {entity['silver_salience']:.2f}")

    # Save the final dataset
    df = pd.DataFrame(salience_matrix)
    output_file = "gum_dev_salience_probing.csv"
    df.to_csv(output_file, index=False)
    print(f"\nBatch processing complete. Dataset saved to {output_file}")

    return df


# ==========================================
# Run the Batch Process
# ==========================================
if __name__ == "__main__":
    # Point this to your folder containing the GUM dev TSV files
    dev_folder_path = "data/"

    results_df = batch_process_dev_set(dev_folder_path, window_size=3)

    print("\nDataset Summary:")
    print(results_df['dep_role'].value_counts())