"""
OmniRAG — Comprehensive Evaluation Script
"""

import sys
import time
import json
import gc
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.evaluation.retrieval_metrics import (
    recall_at_k,
    reciprocal_rank
)

from src.evaluation.faithfulness import faithfulness_score


# =====================================================
# UPDATED EVAL SETS (uses uploaded files)
# =====================================================

EVAL_SETS = {

    "text":{
        "data_path":"outputs/uploads",
        "queries":[
            {
                "question":"Summarize the uploaded text",
                "relevant_phrases":[
                    "summary","main","topic"
                ]
            }
        ]
    },

    "pdf":{
        "data_path":"outputs/uploads",
        "queries":[
            {
                "question":"Summarize the uploaded PDF",
                "relevant_phrases":[
                    "document","summary","content"
                ]
            }
        ]
    },

    "image":{
        "data_path":"outputs/uploads",
        "queries":[
            {
                "question":"Describe the uploaded image",
                "relevant_phrases":[
                    "image","object","scene"
                ]
            }
        ]
    },

    "audio":{
        "data_path":"outputs/uploads",
        "queries":[
            {
                "question":"Summarize uploaded audio",
                "relevant_phrases":[
                    "speech","topic","discussion"
                ]
            }
        ]
    },

    "video":{
        "data_path":"outputs/uploads",
        "queries":[
            {
                "question":"Summarize uploaded video",
                "relevant_phrases":[
                    "video","scene","content"
                ]
            }
        ]
    }

}


# =====================================================
# HELPERS
# =====================================================

def free_vram():
    gc.collect()
    torch.cuda.empty_cache()


def ingest_modality(modality,data_path):

    from src.rag_pipeline import RAGPipeline

    t0=time.time()

    if modality=="text":
        from src.ingestion.text_loader import TextLoader
        docs=TextLoader(data_path).load()

    elif modality=="pdf":
        from src.ingestion.pdf_loader import PDFLoader
        docs=PDFLoader(data_path).load()

    elif modality=="image":
        from src.ingestion.image_captioner import ImageCaptioner

        captioner=ImageCaptioner()

        docs=captioner.caption_dir(data_path)

        captioner.unload()

        free_vram()

    elif modality=="audio":

        from src.ingestion.audio_transcriber import AudioTranscriber

        transcriber=AudioTranscriber(
            model_size="small",
            device="cuda"
        )

        docs=transcriber.transcribe(
            data_path
        )

        del transcriber

        free_vram()

    elif modality=="video":

        from src.ingestion.video_processor import VideoProcessor
        from src.ingestion.video_captioner import VideoCaptioner

        processor=VideoProcessor(
            keyframe_interval=2,
            device="cuda"
        )

        transcript_docs,keyframe_images,keyframe_sources=(
            processor.process(data_path)
        )

        processor.unload()

        free_vram()

        captioner=VideoCaptioner()

        keyframe_docs=captioner.caption_frames(
            keyframe_images,
            keyframe_sources
        )

        captioner.unload()

        free_vram()

        docs=transcript_docs+keyframe_docs

    else:
        raise ValueError(
            f"Unknown modality {modality}"
        )


    doc_count=len(docs)

    pipeline=RAGPipeline()

    pipeline.ingest(
        docs,
        source_dir=data_path
    )

    ingestion_time=time.time()-t0

    chunk_count=0

    if pipeline.text_vectorstore is not None:

        try:
            chunk_count=len(
                pipeline.text_vectorstore.metadata
            )

        except:
            chunk_count=-1


    return (
        pipeline,
        ingestion_time,
        doc_count,
        chunk_count
    )



# =====================================================
# EVALUATION
# =====================================================

def evaluate_pipeline(
    pipeline,
    queries,
    top_k=10
):

    from src.retrieval.text_retriever import TextRetriever

    latencies=[]
    recall_scores=[]
    mrr_scores=[]
    faithfulness_scores=[]
    hallucination_scores=[]


    for item in queries:

        question=item["question"]

        relevant_phrases=item[
            "relevant_phrases"
        ]

        t0=time.time()

        answer=pipeline.query(
            question
        )

        query_latency=time.time()-t0

        latencies.append(
            query_latency
        )


        if pipeline.text_vectorstore is not None:

            retriever=TextRetriever(
                pipeline.text_embedder,
                pipeline.text_vectorstore,
                top_k
            )

            retrieved=retriever.retrieve(
                question
            )

            chunks=[
                r["text"]
                for r in retrieved
            ]

        else:
            chunks=[]


        recall=recall_at_k(
            chunks,
            relevant_phrases,
            k=top_k
        )

        recall_scores.append(
            recall
        )


        rr=reciprocal_rank(
            chunks,
            relevant_phrases
        )

        mrr_scores.append(
            rr
        )


        faith=faithfulness_score(
            answer,
            chunks[:5]
        )

        faithfulness_scores.append(
            faith
        )


        hallucination=1-faith

        hallucination_scores.append(
            hallucination
        )


        print(f"\nQ: {question}")

        print(
            f"Lat={query_latency:.2f}s "
            f"Recall={recall:.2f} "
            f"MRR={rr:.2f} "
            f"Faith={faith:.3f} "
            f"Hall={hallucination:.3f}"
        )


    return {

        "avg_latency":round(
            sum(latencies)/len(latencies),
            2
        ),

        "max_latency":round(
            max(latencies),
            2
        ),

        "avg_recall":round(
            sum(recall_scores)/
            len(recall_scores),
            3
        ),

        "avg_mrr":round(
            sum(mrr_scores)/
            len(mrr_scores),
            3
        ),

        "avg_faithfulness":round(
            sum(faithfulness_scores)/
            len(faithfulness_scores),
            4
        ),

        "hallucination_rate":round(
            sum(hallucination_scores)/
            len(hallucination_scores),
            4
        ),

        "n_queries":len(queries)
    }



# =====================================================
# MAIN
# =====================================================

def main():

    top_k=10

    results={}


    # CHANGED:
    modalities_to_run=list(
        EVAL_SETS.keys()
    )


    for modality in modalities_to_run:

        cfg=EVAL_SETS[modality]

        print(
            f"\nRunning {modality.upper()}"
        )

        try:

            (
                pipeline,
                ingest_time,
                doc_count,
                chunk_count
            )=ingest_modality(
                modality,
                cfg["data_path"]
            )


            metrics=evaluate_pipeline(
                pipeline,
                cfg["queries"],
                top_k
            )


            results[modality]={

                "ingestion_time_s":round(
                    ingest_time,
                    1
                ),

                "doc_count":doc_count,

                "chunk_count":chunk_count,

                **metrics
            }


            del pipeline

            free_vram()

        except Exception as e:
            print(e)



    print(f"\n\n{'='*70}")
    print(" RESULTS SUMMARY")
    print(f"{'='*70}")

    print(
        f"{'Modality':<10}"
        f"{'Recall':<10}"
        f"{'MRR':<10}"
        f"{'Faith':<12}"
        f"{'Halluc':<12}"
        f"{'Latency'}"
    )

    print(f"{'-'*70}")


    for modality in modalities_to_run:

        r=results.get(
            modality,
            {}
        )

        if "error" in r:
            continue


        print(
            f"{modality:<10}"
            f"{r['avg_recall']:<10}"
            f"{r['avg_mrr']:<10}"
            f"{r['avg_faithfulness']:<12}"
            f"{r['hallucination_rate']:<12}"
            f"{r['avg_latency']}"
        )


    print(f"{'='*70}\n")


    out_path=Path(
        "outputs/eval_results.json"
    )

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    with open(out_path,"w") as f:

        json.dump(
            results,
            f,
            indent=2
        )


if __name__=="__main__":
    main()