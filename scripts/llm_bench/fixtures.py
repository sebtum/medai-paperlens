"""Static question + evidence fixtures for benchmark_llm.py and eval_llm.py.

Evidence is hand-picked directly from data/papers/*.json so both scripts can
exercise the real generate/rewrite prompts without needing a live Qdrant
instance or the embedding model - retrieval is deliberately faked here.
"""

from dataclasses import dataclass

from app.models.query import Citation


@dataclass(frozen=True)
class FixtureQuestion:
    id: str
    question: str
    citations: list[Citation]


FIXTURE_QUESTIONS: list[FixtureQuestion] = [
    FixtureQuestion(
        id="llava_med_cost_efficient_training",
        question=(
            "How can a biomedical vision-language conversational assistant be "
            "trained cost-efficiently for open-ended visual question answering?"
        ),
        citations=[
            Citation(
                title="LLaVA-Med: Training a Large Language-and-Vision Assistant "
                "for Biomedicine in One Day",
                authors=["Li, Chunyuan", "Wong, Cliff", "Zhang, Sheng"],
                year=2023,
                doi="10.48550/arXiv.2306.00890",
                excerpt=(
                    "We propose a cost-efficient approach for training a "
                    "vision-language conversational assistant that can answer "
                    "open-ended research questions of biomedical images, using a "
                    "large-scale biomedical figure-caption dataset and GPT-4 "
                    "self-instruct data, trained in under 15 hours on eight A100s."
                ),
            ),
            Citation(
                title="Towards Generalist Biomedical AI",
                authors=["Tu, Tao", "Azizi, Shekoofeh", "Driess, Danny"],
                year=2023,
                doi="10.48550/arXiv.2307.14334",
                excerpt=(
                    "Med-PaLM M is a large multimodal generative model that "
                    "flexibly encodes and interprets biomedical data including "
                    "clinical language, imaging, and genomics with the same set "
                    "of model weights."
                ),
            ),
        ],
    ),
    FixtureQuestion(
        id="med_palm_m_benchmarks",
        question=(
            "What benchmarks exist for evaluating generalist multimodal "
            "biomedical AI systems across imaging, text, and genomics?"
        ),
        citations=[
            Citation(
                title="Towards Generalist Biomedical AI",
                authors=["Tu, Tao", "Azizi, Shekoofeh", "Driess, Danny"],
                year=2023,
                doi="10.48550/arXiv.2307.14334",
                excerpt=(
                    "We curate MultiMedBench, a new multimodal biomedical "
                    "benchmark encompassing 14 diverse tasks such as medical "
                    "question answering, mammography and dermatology image "
                    "interpretation, radiology report generation and "
                    "summarization, and genomic variant calling."
                ),
            ),
            Citation(
                title="LLaVA-Med: Training a Large Language-and-Vision Assistant "
                "for Biomedicine in One Day",
                authors=["Li, Chunyuan", "Wong, Cliff", "Zhang, Sheng"],
                year=2023,
                doi="10.48550/arXiv.2306.00890",
                excerpt=(
                    "On three standard biomedical visual question answering "
                    "datasets, LLaVA-Med outperforms previous supervised "
                    "state-of-the-art on certain metrics."
                ),
            ),
        ],
    ),
    FixtureQuestion(
        id="maira2_grounded_report_generation",
        question=(
            "How can grounded findings localization improve automated chest "
            "X-ray report generation?"
        ),
        citations=[
            Citation(
                title="MAIRA-2: Grounded Radiology Report Generation",
                authors=["Bannur, Shruthi", "Bouzid, Kenza", "Castro, Daniel C."],
                year=2024,
                doi="10.48550/arXiv.2406.04449",
                excerpt=(
                    "We augment the utility of automated report generation by "
                    "incorporating localisation of individual findings on the "
                    "image - a task we call grounded report generation - and "
                    "develop MAIRA-2, a large radiology-specific multimodal "
                    "model designed to generate chest X-ray reports with and "
                    "without grounding."
                ),
            ),
            Citation(
                title="A vision-language foundation model for the generation of "
                "realistic chest X-ray images",
                authors=["Bluethgen, Christian", "Chambon, Pierre"],
                year=2024,
                doi="10.1038/s41551-024-01246-y",
                excerpt=(
                    "We adapted a latent diffusion model pre-trained on pairs of "
                    "natural images and text descriptors to generate diverse and "
                    "visually plausible synthetic chest X-ray images whose "
                    "appearance can be controlled with free-form medical text "
                    "prompts."
                ),
            ),
        ],
    ),
    FixtureQuestion(
        id="radfm_generalist_radiology_foundation_model",
        question=(
            "What large-scale datasets and architectures support a generalist "
            "radiology foundation model spanning 2D and 3D scans?"
        ),
        citations=[
            Citation(
                title="Towards Generalist Foundation Model for Radiology by "
                "Leveraging Web-scale 2D and 3D Medical Data",
                authors=["Wu, Chaoyi", "Zhang, Xiaoman", "Zhang, Ya"],
                year=2023,
                doi="10.48550/arXiv.2308.02463",
                excerpt=(
                    "We construct MedMD, a large-scale Medical Multi-modal "
                    "Dataset of 16M 2D and 3D medical scans with high-quality "
                    "text descriptions across various modalities and tasks, "
                    "covering over 5000 distinct diseases, and propose RadFM, "
                    "an architecture enabling visually conditioned generative "
                    "pre-training on 2D or 3D medical scans."
                ),
            ),
            Citation(
                title="Are Vision Language Models Ready for Clinical Diagnosis? "
                "A 3D Medical Benchmark for Tumor-centric Visual Question "
                "Answering",
                authors=["Chen, Yixiong", "Xiao, Wenjie", "Bassi, Pedro R. A. S."],
                year=2025,
                doi="10.48550/arXiv.2505.18915",
                excerpt=(
                    "Large-scale multimodal pretraining plays a crucial role in "
                    "DeepTumorVQA testing performance, making RadFM stand out "
                    "among all benchmarked vision-language models."
                ),
            ),
        ],
    ),
    FixtureQuestion(
        id="cxr_synthetic_image_generation",
        question=(
            "How can vision-language models be adapted to generate realistic "
            "synthetic chest X-ray images for data augmentation?"
        ),
        citations=[
            Citation(
                title="A vision-language foundation model for the generation of "
                "realistic chest X-ray images",
                authors=["Bluethgen, Christian", "Chambon, Pierre"],
                year=2024,
                doi="10.1038/s41551-024-01246-y",
                excerpt=(
                    "By leveraging publicly available datasets of chest X-ray "
                    "images and corresponding radiology reports, we adapted a "
                    "latent diffusion model pre-trained on natural images to "
                    "overcome the distributional shift towards medical imaging, "
                    "as confirmed by board-certified radiologists."
                ),
            ),
            Citation(
                title="MAIRA-2: Grounded Radiology Report Generation",
                authors=["Bannur, Shruthi", "Bouzid, Kenza", "Castro, Daniel C."],
                year=2024,
                doi="10.48550/arXiv.2406.04449",
                excerpt=(
                    "MAIRA-2 achieves state of the art on existing chest X-ray "
                    "report generation benchmarks and establishes the novel "
                    "task of grounded report generation."
                ),
            ),
        ],
    ),
    FixtureQuestion(
        id="gaze2report_gaze_guided_generation",
        question=(
            "How can radiologists' eye-gaze patterns be incorporated into "
            "automated report generation without requiring gaze data at "
            "inference time?"
        ),
        citations=[
            Citation(
                title="Gaze2Report: Radiology Report Generation via Visual-Gaze "
                "Prompt Tuning of LLMs",
                authors=["Konwer, Aishik", "Bhattacharya, Moinak", "Prasanna, Prateek"],
                year=2026,
                doi="10.48550/arXiv.2604.08600",
                excerpt=(
                    "Gaze2Report leverages a scanpath prediction module and "
                    "Graph Neural Network to generate joint visual-gaze tokens "
                    "used to fine-tune LoRA layers of large language models, "
                    "incorporating on-the-fly scanpath prediction so the model "
                    "can operate without gaze input during inference."
                ),
            ),
            Citation(
                title="MAIRA-2: Grounded Radiology Report Generation",
                authors=["Bannur, Shruthi", "Bouzid, Kenza", "Castro, Daniel C."],
                year=2024,
                doi="10.48550/arXiv.2406.04449",
                excerpt=(
                    "We enhance performance by incorporating realistic "
                    "reporting context as inputs to the report generation "
                    "model."
                ),
            ),
        ],
    ),
    FixtureQuestion(
        id="pathalign_wsi_report_pairing",
        question=(
            "How can whole slide histopathology images be paired with "
            "pathology report text for vision-language pretraining?"
        ),
        citations=[
            Citation(
                title="PathAlign: A vision-language model for whole slide "
                "images in histopathology",
                authors=["Ahmed, Faruk", "Sellergren, Andrew", "Yang, Lin"],
                year=2024,
                doi="10.48550/arXiv.2406.19578",
                excerpt=(
                    "We develop a vision-language model based on the BLIP-2 "
                    "framework using whole slide images paired with curated "
                    "text from pathology reports, using a de-identified "
                    "dataset of over 350,000 WSI and diagnostic text pairs."
                ),
            ),
            Citation(
                title="Towards a Visual-Language Foundation Model for "
                "Computational Pathology",
                authors=["Lu, Ming Y.", "Chen, Bowen", "Williamson, Drew F. K."],
                year=2023,
                doi="10.1038/s41591-024-02856-4",
                excerpt=(
                    "CONCH is a vision-language foundation model for "
                    "computational pathology developed using diverse sources of "
                    "histopathology images, biomedical text, and over 1.17 "
                    "million image-caption pairs via task-agnostic pretraining."
                ),
            ),
        ],
    ),
    FixtureQuestion(
        id="conch_contrastive_pathology_pretraining",
        question=(
            "What contrastive learning approaches enable a general-purpose "
            "vision-language foundation model for computational pathology?"
        ),
        citations=[
            Citation(
                title="Towards a Visual-Language Foundation Model for "
                "Computational Pathology",
                authors=["Lu, Ming Y.", "Chen, Bowen", "Williamson, Drew F. K."],
                year=2023,
                doi="10.1038/s41591-024-02856-4",
                excerpt=(
                    "Evaluated on a suite of 14 diverse benchmarks ranging from "
                    "patch-level to slide-level tasks including classification, "
                    "segmentation, captioning, and text-to-image/image-to-text "
                    "retrieval, CONCH represents a substantial improvement over "
                    "prior art with minimal or no further supervised "
                    "fine-tuning."
                ),
            ),
            Citation(
                title="PathAlign: A vision-language model for whole slide "
                "images in histopathology",
                authors=["Ahmed, Faruk", "Sellergren, Andrew", "Yang, Lin"],
                year=2024,
                doi="10.48550/arXiv.2406.19578",
                excerpt=(
                    "Model-generated text for whole slide images was rated by "
                    "pathologists as accurate, without clinically significant "
                    "error or omission, for 78% of WSIs on average."
                ),
            ),
        ],
    ),
    FixtureQuestion(
        id="deep_tumor_vqa_clinical_readiness",
        question=(
            "How well do current vision-language models perform on 3D "
            "tumor-centric visual question answering compared to clinical "
            "needs?"
        ),
        citations=[
            Citation(
                title="Are Vision Language Models Ready for Clinical Diagnosis? "
                "A 3D Medical Benchmark for Tumor-centric Visual Question "
                "Answering",
                authors=["Chen, Yixiong", "Xiao, Wenjie", "Bassi, Pedro R. A. S."],
                year=2025,
                doi="10.48550/arXiv.2505.18915",
                excerpt=(
                    "Benchmarking four advanced vision-language models on "
                    "DeepTumorVQA, we find current models perform adequately "
                    "on measurement tasks but struggle with lesion recognition "
                    "and reasoning, and are still not meeting clinical needs."
                ),
            ),
            Citation(
                title="Towards Generalist Foundation Model for Radiology by "
                "Leveraging Web-scale 2D and 3D Medical Data",
                authors=["Wu, Chaoyi", "Zhang, Xiaoman", "Zhang, Ya"],
                year=2023,
                doi="10.48550/arXiv.2308.02463",
                excerpt=(
                    "RadBench comprises five tasks including modality "
                    "recognition, disease diagnosis, visual question "
                    "answering, report generation and rationale diagnosis, "
                    "aiming to comprehensively assess foundation models on "
                    "practical clinical problems."
                ),
            ),
        ],
    ),
    FixtureQuestion(
        id="intersectional_fairness_vlm_debiasing",
        question=(
            "What training approaches reduce intersectional bias in "
            "vision-language models used for medical image disease "
            "classification?"
        ),
        citations=[
            Citation(
                title="Intersectional Fairness in Vision-Language Models for "
                "Medical Image Disease Classification",
                authors=["Zhang, Yupeng", "Dunn, Adam G.", "Naseem, Usman"],
                year=2025,
                doi="10.48550/arXiv.2512.15249",
                excerpt=(
                    "Cross-Modal Alignment Consistency (CMAC-MMD) standardises "
                    "diagnostic certainty across intersectional patient "
                    "subgroups without requiring sensitive demographic data "
                    "during clinical inference, reducing the missed diagnosis "
                    "gap while improving overall AUC in dermatology and "
                    "glaucoma screening cohorts."
                ),
            ),
            Citation(
                title="Towards Generalist Biomedical AI",
                authors=["Tu, Tao", "Azizi, Shekoofeh", "Driess, Danny"],
                year=2023,
                doi="10.48550/arXiv.2307.14334",
                excerpt=(
                    "Considerable work is needed to validate generalist "
                    "biomedical AI models in real-world use cases before "
                    "clinical utility can be established."
                ),
            ),
        ],
    ),
    FixtureQuestion(
        id="broadest_downstream_task_coverage",
        question=(
            "Which vision-language foundation models have been evaluated "
            "across the widest range of downstream clinical tasks such as "
            "classification, retrieval, and generation?"
        ),
        citations=[
            Citation(
                title="Towards a Visual-Language Foundation Model for "
                "Computational Pathology",
                authors=["Lu, Ming Y.", "Chen, Bowen", "Williamson, Drew F. K."],
                year=2023,
                doi="10.1038/s41591-024-02856-4",
                excerpt=(
                    "CONCH is evaluated on 14 diverse benchmarks ranging from "
                    "patch-level to slide-level tasks including "
                    "classification, segmentation, captioning, and "
                    "text-to-image/image-to-text retrieval."
                ),
            ),
            Citation(
                title="Towards Generalist Biomedical AI",
                authors=["Tu, Tao", "Azizi, Shekoofeh", "Driess, Danny"],
                year=2023,
                doi="10.48550/arXiv.2307.14334",
                excerpt=(
                    "Med-PaLM M reaches performance competitive with or "
                    "exceeding the state of the art on all 14 MultiMedBench "
                    "tasks, often surpassing specialist models by a wide "
                    "margin."
                ),
            ),
            Citation(
                title="Towards Generalist Foundation Model for Radiology by "
                "Leveraging Web-scale 2D and 3D Medical Data",
                authors=["Wu, Chaoyi", "Zhang, Xiaoman", "Zhang, Ya"],
                year=2023,
                doi="10.48550/arXiv.2308.02463",
                excerpt=(
                    "RadFM outperforms existing publicly accessible "
                    "multi-modal foundation models across all five RadBench "
                    "tasks in both automatic and human evaluation."
                ),
            ),
        ],
    ),
    FixtureQuestion(
        id="domain_adaptation_general_to_medical_vlm",
        question=(
            "What are the main approaches for adapting general-domain "
            "vision-language models to specialized medical imaging domains "
            "like radiology and pathology?"
        ),
        citations=[
            Citation(
                title="LLaVA-Med: Training a Large Language-and-Vision Assistant "
                "for Biomedicine in One Day",
                authors=["Li, Chunyuan", "Wong, Cliff", "Zhang, Sheng"],
                year=2023,
                doi="10.48550/arXiv.2306.00890",
                excerpt=(
                    "A novel curriculum learning method first aligns "
                    "biomedical vocabulary using figure-caption pairs, then "
                    "masters open-ended conversational semantics using GPT-4 "
                    "generated instruction-following data."
                ),
            ),
            Citation(
                title="A vision-language foundation model for the generation of "
                "realistic chest X-ray images",
                authors=["Bluethgen, Christian", "Chambon, Pierre"],
                year=2024,
                doi="10.1038/s41551-024-01246-y",
                excerpt=(
                    "We describe a domain-adaptation strategy for large "
                    "vision-language models that overcomes the distributional "
                    "shift between natural and medical images."
                ),
            ),
            Citation(
                title="PathAlign: A vision-language model for whole slide "
                "images in histopathology",
                authors=["Ahmed, Faruk", "Sellergren, Andrew", "Yang, Lin"],
                year=2024,
                doi="10.48550/arXiv.2406.19578",
                excerpt=(
                    "We develop a vision-language model based on the BLIP-2 "
                    "framework, adapting it to gigapixel-scale whole slide "
                    "images paired with curated pathology report text."
                ),
            ),
        ],
    ),
]


def get_fixture_questions() -> list[FixtureQuestion]:
    return FIXTURE_QUESTIONS
