import os
import time
import logging
from datetime import datetime, timedelta
from pymed import PubMed
from tqdm import tqdm

# Configuration logging
log_filename = f"logs/pubmed_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Paramètres personnalisables
SEARCH_TERM = "breast cancer"
EMAIL = "your_email@usmba.ac.ma"
START_DATE = datetime(2020, 1, 1)
END_DATE = datetime(2025, 1, 1)
STEP_DAYS = 5
MAX_ABSTRACTS = 200000
OUTPUT_FILE = "breast_cancer.txt"
CHECKPOINT_FILE = "checkpoint.txt"

def daterange(start_date, end_date, step_days):
    current = start_date
    while current <= end_date:
        yield current, min(current + timedelta(days=step_days), end_date)
        current += timedelta(days=step_days + 1)

def extract_abstracts_in_range(search_term, start, end, email):
    pubmed = PubMed(tool="PubMedRetriever", email=email)
    query = f'{search_term} AND ("{start.strftime("%Y/%m/%d")}"[PDAT] : "{end.strftime("%Y/%m/%d")}"[PDAT])'
    logger.info(f"Requête: {query}")

    try:
        results = pubmed.query(query, max_results=10000)
        abstracts = []

        for article in results:
            if article.abstract:
                abstract = article.abstract.encode("ascii", errors="ignore").decode("ascii").replace("\n", " ").replace("\r", " ").strip()
                if abstract:
                    abstracts.append(abstract)

        logger.info(f"{len(abstracts)} abstracts extraits entre {start} et {end}")
        return abstracts

    except Exception as e:
        logger.error(f"Erreur lors de la récupération entre {start} et {end}: {e}")
        return []

def save_checkpoint(date):
    with open(CHECKPOINT_FILE, "w") as f:
        f.write(date.strftime("%Y-%m-%d"))

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return datetime.strptime(f.read().strip(), "%Y-%m-%d")
    return START_DATE

def main():
    os.makedirs("checkpoints", exist_ok=True)

    abstracts = []
    total_count = 0
    current_start = load_checkpoint()

    logger.info("Début de l'extraction...")
    logger.info(f"Recherche: '{SEARCH_TERM}', depuis {current_start.date()} jusqu'à {END_DATE.date()}")

    with open(OUTPUT_FILE, "a", encoding="utf-8") as out:
        for start, end in tqdm(daterange(current_start, END_DATE, STEP_DAYS), desc="Téléchargement des abstracts"):
            time.sleep(2)
            batch = extract_abstracts_in_range(SEARCH_TERM, start, end, EMAIL)

            for abstract in batch:
                out.write(abstract + "\n")

            total_count += len(batch)
            logger.info(f"Progression: {total_count}/{MAX_ABSTRACTS} abstracts")

            save_checkpoint(end + timedelta(days=1))

            # Sauvegarde toutes les 10 000 entrées
            if total_count >= MAX_ABSTRACTS:
                logger.info("Objectif atteint. Arrêt.")
                break

    logger.info(f"Extraction terminée. Total: {total_count} abstracts.")
    logger.info(f"Fichier sauvegardé: {OUTPUT_FILE}")
    print(f"Extraction terminée. Total: {total_count} abstracts.")
    print(f"Fichier sauvegardé: {OUTPUT_FILE}")

if __name__ == "__main__":
        main()