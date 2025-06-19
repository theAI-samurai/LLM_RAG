import os
import json
import pandas as pd
import MeCab
from jiwer import wer, cer
from langchain_community.embeddings import OCIGenAIEmbeddings
# from transformers.models.align.convert_align_tf_to_hf import preprocess
# from unstructured_inference.models.yolox import preprocess

from environment import AUTH_TYPE, REGION, OCI_PROFILE, MODEL_ID, GENAI_ENDPOINT,COMPARTMENT_ID
from environment import ORACLE_DB_CONNECTION_STRING, ORACLE_DB_USER, ORACLE_DB_PASSWORD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import  oracledb


def tokenize_japanese(text):
    mecab = MeCab.Tagger()  # Wakati mode (space-separated words)
    tokenized = mecab.parse(text).strip()
    return tokenized


def japanese_wer(reference, hypothesis):                #   word error rate
    ref_tokenized = tokenize_japanese(reference)
    hyp_tokenized = tokenize_japanese(hypothesis)
    return wer(ref_tokenized, hyp_tokenized)


def tokenize_apply(df_, colname):
    toke = []
    for i in range(len(df_)):
        tokeni = tokenize_japanese(df_.loc[i, colname])
        toke.append(tokeni)
    return toke


def japanese_cer(reference, hypothesis):
    return cer(reference, hypothesis)


def get_embedder(language):
    """
    The get_embedder method initializes an instance of OCIGenAIEmbeddings with specific model parameters such as model_id, service_endpoint, compartment_id, auth_type, and model_kwargs.
    It returns the initialized OCIGenAIEmbeddings object.
    """
    if language == 'en':
        model_id = "cohere.embed-english-v3.0"
    elif language == 'jp':
        model_id = "cohere.embed-multilingual-v3.0"

    embedder = OCIGenAIEmbeddings(
        model_id=model_id,
        service_endpoint=GENAI_ENDPOINT,
        compartment_id=COMPARTMENT_ID,
        auth_type=AUTH_TYPE,
        auth_profile=OCI_PROFILE,
        model_kwargs={"input_type": "SEARCH_DOCUMENT"}
    )
    return embedder

def preptext(text):
    text = text.replace('橋本', '').replace('中川', '').replace('「', '').replace('：','').replace('」', '')
    text = text.strip()
    return text

embedder = get_embedder('jp')

refDF = pd.read_excel('Transcripts.xlsx', sheet_name='Sample1 JP')
refDF = refDF.dropna()
refDF.reset_index(drop=True, inplace=True)

transdf = pd.read_csv('/Users/anankitm/Documents/NRI_audio/transcripts_converted/whisper_medium/conversation_sample_special_benefit_provision_converted_conversation.txt', sep=':', header=None)


cerlst = []
cos_sim = []
ref_txt = []
gen_txt = []
ref_embed = []
gen_embed = []
for i in range(len(transdf)):
    ref = preptext(transdf.loc[i, 1])
    hyp = preptext(refDF.loc[i, 'Speaker : text'])
    cerrate = cer(ref, hyp)
    print(f"Character Error Rate (CER): {cerrate:.2f}")
    cerlst.append(cerrate)

    res_ref = embedder.embed_documents([ref])
    res_hyp = embedder.embed_documents([hyp])
    similarity = cosine_similarity(res_ref, res_hyp)
    cos_sim.append(similarity[0][0])

    ref_txt.append(ref)
    gen_txt.append(hyp)

    ref_embed.append(res_ref)
    gen_embed.append(res_hyp)

padded_cer = cerlst + [np.nan] * (len(refDF) - len(cerlst))
padded_sim = cos_sim + [np.nan] * (len(refDF) - len(cos_sim))
padded_ref = ref_txt + [np.nan] * (len(refDF) - len(ref_txt))
padded_gen = gen_txt + [np.nan] * (len(refDF) - len(gen_txt))
padded_refem = ref_embed + [np.nan] * (len(refDF) - len(ref_embed))
padded_genem = gen_embed + [np.nan] * (len(refDF) - len(gen_embed))

refDF['cer'] = padded_cer
refDF['embed_sim'] = padded_sim
refDF['reftxt'] = padded_ref
refDF['gentxt'] = padded_gen

refDF['gn_embedding'] = padded_refem
refDF['trans_embedding'] = padded_genem
refDF['gn_embedding'] = refDF['gn_embedding'].apply(json.dumps)
refDF['trans_embedding'] = refDF['trans_embedding'].apply(json.dumps)

refDF['gentxt'] = refDF['gentxt'].fillna("")
refDF['reftxt'] = refDF['reftxt'].fillna("")

refDF['cer'] = refDF['cer'].fillna(0)
refDF['embed_sim'] = refDF['embed_sim'].fillna(0)


username = "ADMIN"
password = "*Nriociuc0123"
wallet_location = "/Users/anankitm/Documents/NRI_audio/Wallet_nrispeechdb-2"
os.environ["TNS_ADMIN"] = wallet_location
dsn = "nrispeechdb_high"
connection = oracledb.connect(
    user=username, password=password, dsn=dsn,
    config_dir=wallet_location,  # Directory containing tnsnames.ora
    wallet_location=wallet_location,
    wallet_password= "@nkiT2829" # Directory containing ewallet.pem
)
print("Connected to Oracle Autonomous Database!")
# cursor = connection.cursor()


rows = [tuple(x) for x in refDF.to_numpy()]


def insert_rows_one_by_one(connection, rows):
    from tqdm import tqdm
    """Insert rows individually with execute()"""
    cursor = connection.cursor()
    success_count = 0

    for row in tqdm(rows, desc="Inserting rows"):
        try:
            # Prepare data - convert lists to JSON strings
            processed_row = (
                row[0],  # RAW_TEXT
                row[1],  # CER
                row[2],  # COSINE_SIM
                row[3],  # GT_TEXT
                row[4],  # TRANSCRIBED_TEXT
                json.dumps(row[5]) if isinstance(row[5], list) else row[5],  # GT_EMBED
                json.dumps(row[6]) if isinstance(row[6], list) else row[6]  # TRANSCRIBED_EMBED
            )

            cursor.execute("""
                INSERT INTO NRI_SPEECH_EMBEDDING_INGESTION 
                (RAW_TEXT, CER, COSINE_SIM, GT_TEXT, TRANSCRIBED_TEXT, GT_EMBED, TRANSCRIBED_EMBED)
                VALUES (:1, :2, :3, :4, :5, :6, :7)
            """, processed_row)

            success_count += 1

            # Commit every 100 rows to balance performance and safety
            connection.commit()

        except Exception as e:
            print(f"\nError inserting row: {row[:5]}...")  # Show partial row to avoid huge output
            print(f"Error details: {str(e)}")
            connection.rollback()  # Only rollback the current failed row

    # Final commit for any remaining rows
    connection.commit()
    cursor.close()
    return success_count

inserted_count = insert_rows_one_by_one(connection, rows)
print(f"Successfully inserted {inserted_count}/{len(rows)} rows")
connection.close()

