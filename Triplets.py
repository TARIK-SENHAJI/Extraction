import spacy
import itertools
import re
from typing import List, Tuple, Optional, Set, Dict
import cupy  # For GPU acceleration with spaCy

class BiomedicalRelationExtractor:

    def __init__(self, use_gpu: bool = True):

        # Set up GPU acceleration if available and requested
        if use_gpu:
            try:
                # Enable GPU acceleration for spaCy
                spacy.require_gpu()
                self.using_gpu = True
                print("GPU acceleration enabled for spaCy")
            except Exception as e:
                print(f"Could not enable GPU acceleration: {e}")
                self.using_gpu = False
        else:
            self.using_gpu = False

        # Load SpaCy models for NER and linguistic analysis
        try:
            # Biomedical NER model
            self.ner_model = spacy.load("en_ner_bionlp13cg_md")
        except OSError:
            print("Biomedical NER model not found. Using general English model instead.")
            self.ner_model = spacy.load("en_core_web_sm")

        # Load general linguistic model for syntax analysis
        self.nlp = spacy.load("en_core_web_sm")

        # Enable batch processing for better GPU utilization
        if self.using_gpu:
            self.ner_model.batch_size = 128
            self.nlp.batch_size = 128

        # Define valid relationship verbs for biomedical domain
        self.valid_verbs: Set[str] = {
            # Treatment/therapy related
            "treat", "prevent", "cure", "manage", "alleviate", "relieve",
            # Molecular interactions
            "bind", "interact", "activate", "inhibit", "block", "suppress",
            "stimulate", "induce", "regulate", "modulate", "mediate",
            # Effect relationships
            "cause", "reduce", "increase", "decrease", "enhance", "promote",
            "impair", "improve", "worsen", "affect", "influence", "target"
        }

        # Common patterns for relation extraction
        self.relation_patterns = [
                # Direct verb patterns
                (r'{}\s+(\w+)s?\s+.*?{}'.format, 1),  # "Drug inhibits Protein"
                (r'{}\s+can\s+(\w+)\s+.*?{}'.format, 1),  # "Drug can treat Disease"

                # Passive constructions
                (r'{}\s+.*?(?:is|are|was|were)\s+(\w+)ed\s+by\s+.*?{}'.format, 1),  # "Protein is inhibited by Drug"
                (r'{}\s+.*?by\s+(\w+)ing\s+.*?{}'.format, 1),  # "Protein is targeted by binding Drug"

                # Therapeutic / medication
                (r'{}\s+(?:is|was)?\s*(?:an|a)?\s*(?:treatment|therapy|medication)\s+for\s+{}'.format, 0, "treats"),
                (r'{}\s+(?:administered|prescribed|recommended)\s+for\s+{}'.format, 1, "treats"),
                (r'{}\s+helps\s+(?:to\s+)?(\w+)\s+{}'.format, 1),  # "Drug helps to control Disease"

                # Diagnostic
                (r'{}\s+(?:is|was)?\s+used\s+to\s+(detect|diagnose|identify)\s+{}'.format, 1, "diagnoses"),
                (r'{}\s+reveals\s+{}'.format, 0, "diagnoses"),
                (r'{}\s+detects\s+{}'.format, 0, "diagnoses"),

                # Cause and effect
                (r'{}\s+(?:cause|causes|lead|leads|result|results)\s+(?:to|in)?\s+.*?{}'.format, 0, "cause"),
                (r'{}\s+(?:trigger|triggers|induce|induces|promote|promotes)\s+{}'.format, 0, "induces"),

                # Genetic mutations
                (r'{}\s+(?:mutation|variant|alteration)s?\s+(?:in|of)\s+{}'.format, 0, "mutated_in"),
                (r'{}\s+is\s+(?:a|an)?\s*(?:mutation|variant)\s+of\s+{}'.format, 0, "mutation_of"),

                # Biomarkers
                (r'{}\s+is\s+(?:a|an)?\s*(?:biomarker|indicator|predictor)\s+for\s+{}'.format, 0, "biomarker_for"),

                # Risk factors
                (r'{}\s+is\s+(?:a|an)?\s*(?:risk\s+factor|predisposing\s+factor)\s+for\s+{}'.format, 0, "risk_factor_for"),

                # Associations
                (r'{}\s+(?:is|was)?\s*(?:associated|linked|correlated)\s+with\s+{}'.format, 0, "associated_with"),
                (r'{}\s+has\s+a\s+significant\s+association\s+with\s+{}'.format, 0, "associated_with"),
                (r'{}\s+is\s+positively\s+correlated\s+with\s+{}'.format, 0, "associated_with"),

                # Symptoms and clinical features
                (r'{}\s+(?:is|are)?\s*(?:a|an)?\s*symptom\s+of\s+{}'.format, 0, "symptom_of"),
                (r'{}\s+occurs\s+in\s+{}'.format, 0, "symptom_of"),
                (r'{}\s+is\s+observed\s+in\s+{}'.format, 0, "symptom_of"),
                (r'{}\s+is\s+seen\s+in\s+{}'.format, 0, "symptom_of"),

                # Preventive actions
                (r'{}\s+(?:prevent|prevents|reduces|mitigates)\s+the\s+risk\s+of\s+{}'.format, 0, "prevents"),
                (r'{}\s+protects\s+against\s+{}'.format, 0, "protects"),

                # Enhances or inhibits
                (r'{}\s+(?:enhances|stimulates|activates)\s+{}'.format, 0, "activates"),
                (r'{}\s+(?:inhibits|suppresses|blocks)\s+{}'.format, 0, "inhibits"),

                # Pathways and processes
                (r'{}\s+is\s+(?:part\s+of|involved\s+in)\s+the\s+pathway\s+of\s+{}'.format, 0, "involved_in"),
                (r'{}\s+plays\s+a\s+role\s+in\s+{}'.format, 0, "involved_in"),

                # Expression levels
                (r'{}\s+is\s+(?:over|under)?expressed\s+in\s+{}'.format, 0, "expressed_in"),
                (r'{}\s+expression\s+is\s+(?:increased|decreased|upregulated|downregulated)\s+in\s+{}'.format, 0, "expressed_in"),
            ]

        # Precompile patterns for faster execution
        self.compiled_patterns: Dict[str, re.Pattern] = {}

    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """Extract biomedical entities with their types from text."""
        doc = self.ner_model(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]

        # If no entities found with biomedical model, extract noun phrases as fallback
        if not entities:
            doc_general = self.nlp(text)
            # Filter for meaningful noun phrases (length > 1 word or proper nouns)
            entities = [(chunk.text, "NOUN_PHRASE") for chunk in doc_general.noun_chunks
                       if len(chunk.text.split()) > 1 or any(token.pos_ == "PROPN" for token in chunk)]

        return entities

    def normalize_verb(self, verb: str) -> Optional[str]:

        verb = verb.lower().strip()

        # Direct match
        if verb in self.valid_verbs:
            return verb

        # Handle common suffixes
        for base_verb in self.valid_verbs:
            if verb == base_verb + "s" or verb == base_verb + "ed" or verb == base_verb + "ing":
                return base_verb

            # Handle some irregular forms
            if verb == "bound" and base_verb == "bind":
                return "bind"
            if verb == "caused" and base_verb == "cause":
                return "cause"

        # Check for verb in word (for compound words or typos)
        for base_verb in self.valid_verbs:
            if base_verb in verb and len(base_verb) > 3:  # Only consider substantial matches
                return base_verb

        return None

    def extract_relation_from_dependency_parse(self, text: str, entity1: str, entity2: str) -> Optional[str]:

        doc = self.nlp(text)

        # Find spans corresponding to entities
        entity1_tokens = set()
        entity2_tokens = set()

        for token in doc:
            if entity1.lower() in token.text.lower():
                entity1_tokens.add(token)
            if entity2.lower() in token.text.lower():
                entity2_tokens.add(token)

        # If entities weren't found as exact tokens, find tokens within the entity spans
        if not entity1_tokens or not entity2_tokens:
            for token in doc:
                if token.text.lower() in entity1.lower():
                    entity1_tokens.add(token)
                if token.text.lower() in entity2.lower():
                    entity2_tokens.add(token)

        # Look for verbs connecting the entities
        for token in doc:
            # Check if token is a verb
            if token.pos_ == "VERB":
                # Check if this verb connects our entities
                e1_connected = any(self._are_connected(token, e1_token) for e1_token in entity1_tokens)
                e2_connected = any(self._are_connected(token, e2_token) for e2_token in entity2_tokens)

                if e1_connected and e2_connected:
                    normalized = self.normalize_verb(token.lemma_)
                    if normalized:
                        return normalized

        return None

    def _are_connected(self, token1, token2):

        # Check direct connection
        if token1.head == token2 or token2.head == token1:
            return True

        # Check if they share a head
        if token1.head == token2.head and token1.head != token1:
            return True

        # Follow dependency chain (limit depth to avoid cycles)
        visited = set()
        current = token1
        for _ in range(5):  # Limit depth
            if current in visited:
                break
            visited.add(current)
            if current == token2:
                return True
            current = current.head

        return False

    def extract_relation_with_patterns(self, text: str, entity1: str, entity2: str) -> Optional[str]:


        for pattern_func, group_idx, *default_verb in self.relation_patterns:
            # Generate the pattern by injecting the entities with re.escape
            pattern_str = pattern_func(re.escape(entity1), re.escape(entity2))

            # Check if the compiled pattern is already in our dictionary
            if pattern_str in self.compiled_patterns:
                pattern = self.compiled_patterns[pattern_str]
            else:
                pattern = re.compile(pattern_str, re.IGNORECASE)
                self.compiled_patterns[pattern_str] = pattern

            matches = pattern.search(text)
            if matches:
                # If the pattern specifies a default verb, use it
                if default_verb:
                    return default_verb[0]

                # Otherwise, extract the verb from the captured group
                try:
                    verb = matches.group(group_idx).lower()
                    normalized = self.normalize_verb(verb)
                    if normalized:
                        return normalized
                except IndexError:
                    # Handle case where the group isn't found
                    continue

        # Try the reversed direction (entity2 -> entity1)
        for pattern_func, group_idx, *default_verb in self.relation_patterns:
            pattern_str = pattern_func(re.escape(entity2), re.escape(entity1))
            if pattern_str in self.compiled_patterns:
                pattern = self.compiled_patterns[pattern_str]
            else:
                pattern = re.compile(pattern_str, re.IGNORECASE)
                self.compiled_patterns[pattern_str] = pattern

            matches = pattern.search(text)
            if matches:
                if default_verb:
                    verb = default_verb[0]
                else:
                    try:
                        verb = matches.group(group_idx).lower()
                        verb = self.normalize_verb(verb)
                    except IndexError:
                        continue

                if verb:
                    inverse_map = {
                        "activate": "activated_by",
                        "inhibit": "inhibited_by",
                        "block": "blocked_by",
                        "cause": "caused_by",
                        "treat": "treated_by"
                    }
                    return inverse_map.get(verb, f"reverse_{verb}")

        return None

    def _filter_triplets(self, triplets: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:

        if not triplets:
            return []

        # Group triplets by entity pair
        entity_pair_dict: Dict[Tuple[str, str], List[Tuple[str, str, str]]] = {}
        for e1, rel, e2 in triplets:
            key = (e1.lower(), e2.lower())
            if key not in entity_pair_dict:
                entity_pair_dict[key] = []
            # Store original case triplet
            entity_pair_dict[key].append((e1, rel, e2))

        # Select the most likely relation for each entity pair
        filtered_triplets = []
        for _, triplet_list in entity_pair_dict.items():
            # If there's only one relation, use it
            if len(triplet_list) == 1:
                filtered_triplets.append(triplet_list[0])
            else:
                # Otherwise count relations (preserving original casing)
                relation_counter = {}
                for _, rel, _ in triplet_list:
                    relation_counter[rel] = relation_counter.get(rel, 0) + 1

                # Get the most frequent relation
                best_relation = max(relation_counter.items(), key=lambda x: x[1])[0]

                # Use the first original triplet with this relation
                for triplet in triplet_list:
                    if triplet[1] == best_relation:
                        filtered_triplets.append(triplet)
                        break

        return filtered_triplets

    def predict_relation(self, text: str, entity1: str, entity2: str) -> Optional[str]:

        # Try pattern-based extraction first (faster)
        relation = self.extract_relation_with_patterns(text, entity1, entity2)
        if relation:
            return relation

        # Try dependency parsing
        relation = self.extract_relation_from_dependency_parse(text, entity1, entity2)
        if relation:
            return relation

        return None

    def extract_triplets(self, text: str) -> List[Tuple[str, str, str]]:

        # Extract entities
        entity_info = self.extract_entities(text)
        entity_texts = [entity for entity, _ in entity_info]

        # Store found triplets
        triplets = []

        # Try all entity pairs
        for entity1, entity2 in itertools.permutations(entity_texts, 2):
            # Skip if entities are too similar or one contains the other
            if entity1.lower() in entity2.lower() or entity2.lower() in entity1.lower():
                continue

            # Predict relation
            relation = self.predict_relation(text, entity1, entity2)

            # Add to triplets if relation found
            if relation:
                # Check if it's a "reverse" relation
                if relation.startswith("reverse_"):
                    # Swap entities and use base relation
                    base_relation = relation[8:]  # Remove "reverse_" prefix
                    triplets.append((entity2, base_relation, entity1))
                # Check if it's an inverse relation (e.g., "inhibited_by")
                elif relation.endswith("_by"):
                    # Swap entities and use base relation
                    base_relation = relation[:-3]  # Remove "_by" suffix
                    if base_relation.endswith("ed"):
                        base_relation = base_relation[:-2]  # Remove "ed" suffix
                    triplets.append((entity2, base_relation, entity1))
                else:
                    triplets.append((entity1, relation, entity2))

        # Filter duplicate and contradictory triplets
        return self._filter_triplets(triplets)

    def process_batch(self, texts: List[str]) -> List[List[Tuple[str, str, str]]]:

        results = []

        # Process in batches for NER and dependency parsing
        ner_docs = list(self.ner_model.pipe(texts))
        nlp_docs = list(self.nlp.pipe(texts))

        # Cache the docs for each text
        self.batch_ner_docs = {text: doc for text, doc in zip(texts, ner_docs)}
        self.batch_nlp_docs = {text: doc for text, doc in zip(texts, nlp_docs)}

        # Process each text
        for text in texts:
            triplets = self.extract_triplets(text)
            results.append(triplets)

        # Clear cache
        self.batch_ner_docs = {}
        self.batch_nlp_docs = {}

        return results


if __name__ == "__main__":
    # Initialize the extractor
    extractor = BiomedicalRelationExtractor(use_gpu=True)
    input_file_path = "chemin"
    output_file_path = "pt"
    batch_size = 32  # Process texts in batches for better GPU utilization

    # Read all lines from input file
    with open(input_file_path, 'r') as file:
        all_lines = [line.strip() for line in file if line.strip()]

    # Process in batches
    with open(output_file_path, 'w') as output_file:
        for i in range(0, len(all_lines), batch_size):
            batch = all_lines[i:i+batch_size]
            # Process this batch
            batch_results = extractor.process_batch(batch)

            # Write results to output file
            for text_triplets in batch_results:
                for triplet in text_triplets:
                    output_file.write(f"{triplet}\n")