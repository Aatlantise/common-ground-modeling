import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import stanza
import pandas as pd

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

    # def analyze_salience(self, context, continuations, target_entity_text):
    #     """
    #     Uses Stanza to track entity clusters across context and generated continuations.
    #     """
    #     mentions_count = 0
    #     context_char_len = len(context)
    #     all_competitor_data = []
    #
    #     for continuation in continuations:
    #         if not continuation: continue
    #         full_text = context + [(sent + "").split(" ") for sent in continuation.split(". ")]
    #         doc = self.nlp(full_text)
    #
    #         # Stanza coref clusters are accessed via doc.clusters
    #         target_cluster_id = None
    #
    #         # 1. Identify the cluster ID for our target entity in the context
    #         # We look for a mention that matches our text AND starts within the context
    #         target_chain = None
    #         context_char_len = len(context)
    #
    #         for chain in doc.coref:
    #             # Stanza chains contain a list called 'mentions'
    #             for mention in chain.mentions:
    #
    #                 # 1. Get the correct sentence
    #                 # Depending on the exact sub-version of Stanza, the attribute is 'sent_index' or 'sentence'
    #                 sent_idx = getattr(mention, 'sent_index', getattr(mention, 'sentence', 0))
    #                 sentence = doc.sentences[sent_idx]
    #
    #                 # 2. Extract the actual word objects using the mention's indices
    #                 # Stanza's coref start/end indices align with the sentence.words list
    #                 mention_words = sentence.words[mention.start_word: mention.end_word]
    #
    #                 if not mention_words:
    #                     continue
    #
    #                 # 3. Reconstruct text and get character offsets from the Word objects
    #                 mention_text = " ".join([w.text for w in mention_words])
    #                 start_char = mention_words[0].start_char
    #
    #                 # 4. Apply your matching logic!
    #                 if target_entity_text.lower() in mention_text.lower() and start_char < context_char_len:
    #                     target_chain = chain
    #                     break  # Found our target mention!
    #
    #             if target_chain:
    #                 break  # Break the outer loop since we found the target chain
    #
    #         # 'survived' should be reset to False at the start of every continuation loop
    #         survived = False
    #         current_continuation_competitors = 0
    #
    #         for chain in doc.coref:
    #             # Check if this chain is the one we identified as the target
    #             is_target = (chain == target_chain)
    #
    #             for mention in chain.mentions:
    #                 # 1. Resolve the sentence and word objects
    #                 sent_idx = getattr(mention, 'sent_index', getattr(mention, 'sentence', 0))
    #                 sentence = doc.sentences[sent_idx]
    #
    #                 # Stanza's coref indices are 0-based pointers into sentence.words
    #                 mention_words = sentence.words[mention.start_word: mention.end_word]
    #                 if not mention_words:
    #                     continue
    #
    #                 # 2. Only look at mentions that start in the generated portion
    #                 # We check the start_char of the first word in the mention
    #                 if mention_words[0].start_char >= context_char_len:
    #                     if is_target:
    #                         # If the target is mentioned at least once in this continuation
    #                         if not survived:
    #                             mentions_count += 1
    #                             survived = True
    #                     else:
    #                         # 3. This is a competitor. Extract features using the head_index
    #                         # mention.head_index is the 0-based index of the head word in the sentence
    #                         head_word = sentence.words[mention.start_word]
    #
    #                         all_competitor_data.append({
    #                             "text": " ".join([w.text for w in mention_words]),
    #                             "role": head_word.deprel,  # Universal Dependency (e.g., nsubj, obj)
    #                             "pos": head_word.upos,  # Universal POS (e.g., PRON, NOUN)
    #                             "is_pronoun": head_word.upos == 'PRON'
    #                         })
    #                         current_continuation_competitors += 1
    #
    #     salience_prob = mentions_count / len(continuations)
    #     return {
    #         "salience_probability": salience_prob,
    #         "competitor_mentions": all_competitor_data,
    #         "avg_competitor_pressure": len(all_competitor_data) / len(continuations)
    #     }

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


# ==========================================
# Run the Pilot
# ==========================================
if __name__ == "__main__":
    # Replace with your actual GUM file path
    file_path = "GUM_news_korea.tsv"

    # For this pilot, pick an entity you know is heavily discussed in the document
    target = "North Korea"

    results_df = run_document_pilot(file_path, target, window_size=3)

    # Save the output to a CSV so you can graph the decay curve!
    results_df.to_csv("salience_matrix_pilot.csv", index=False)
    print("\nSimulation complete. Data saved to salience_matrix_pilot.csv")
    print(results_df.head(10))