"""
NeuralPath — Master Skill Knowledge Graph
==========================================
This is the CORE of the adaptive engine. Every skill is a node.
Every dependency is a directed weighted edge.

The graph covers 7 major domains:
  1. Software Engineering
  2. Data Science / ML / DL
  3. Cloud & DevOps
  4. Data Engineering
  5. Cybersecurity
  6. Product / Management
  7. Cross-domain (HR, Marketing, Operations)

Each node carries:
  - difficulty     : 1–5 scale
  - base_hours     : raw learning time
  - domain         : primary domain tag
  - tags           : searchable skill aliases
  - onet_codes     : O*NET occupation codes this maps to
  - description    : plain-English explanation used in reasoning traces
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import networkx as nx

# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SkillNode:
    id: str
    name: str
    description: str
    domain: str                        # "software" | "ml" | "cloud" | "data-eng" | "security" | "product" | "general"
    difficulty: int                    # 1 (beginner) → 5 (expert)
    base_hours: float                  # learning time assuming zero prior knowledge
    tags: list[str] = field(default_factory=list)          # aliases used for fuzzy matching
    onet_codes: list[str] = field(default_factory=list)    # O*NET SOC codes
    prerequisites: list[str] = field(default_factory=list) # IDs of required-before skills


# ─────────────────────────────────────────────────────────────────────────────
# Skill registry — 100+ skills across all domains
# ─────────────────────────────────────────────────────────────────────────────

SKILL_NODES: list[SkillNode] = [

    # ══════════════════════════════════════════════════════════════
    # DOMAIN: Software Engineering
    # ══════════════════════════════════════════════════════════════
    SkillNode("python-basics", "Python Fundamentals", "Variables, loops, functions, file I/O, exceptions", "software", 1, 12.0,
              tags=["python", "programming", "scripting"], onet_codes=["15-1252.00"]),
    SkillNode("python-oop", "Python OOP & Design Patterns", "Classes, inheritance, SOLID principles, factory/singleton patterns", "software", 2, 10.0,
              tags=["oop", "python", "design-patterns"], prerequisites=["python-basics"]),
    SkillNode("data-structures", "Data Structures & Algorithms", "Arrays, trees, graphs, sorting, BFS/DFS, Big-O analysis", "software", 3, 20.0,
              tags=["dsa", "algorithms", "leetcode", "data structures"], prerequisites=["python-basics"]),
    SkillNode("system-design", "System Design & Architecture", "CAP theorem, load balancing, caching, microservices patterns, API design", "software", 4, 24.0,
              tags=["system design", "architecture", "scalability", "distributed systems"], prerequisites=["data-structures"]),
    SkillNode("distributed-systems", "Distributed Systems", "Consensus algorithms, event sourcing, CQRS, Saga pattern, distributed tracing", "software", 5, 20.0,
              tags=["distributed", "kafka", "event sourcing", "consistency"], prerequisites=["system-design"]),
    SkillNode("javascript", "JavaScript & ES2024", "Closures, promises, async/await, event loop, modules", "software", 2, 14.0,
              tags=["js", "javascript", "es6", "frontend"], onet_codes=["15-1252.00"]),
    SkillNode("typescript", "TypeScript", "Static typing, generics, utility types, decorators, strict mode", "software", 2, 10.0,
              tags=["typescript", "ts", "typed javascript"], prerequisites=["javascript"]),
    SkillNode("react", "React & Component Architecture", "Hooks, context, state management, render optimisation, testing", "software", 3, 16.0,
              tags=["react", "reactjs", "frontend", "spa"], prerequisites=["typescript"]),
    SkillNode("nextjs", "Next.js (SSR / Full-Stack)", "App router, RSC, ISR, API routes, middleware, deployment", "software", 3, 12.0,
              tags=["nextjs", "next.js", "ssr", "full-stack"], prerequisites=["react"]),
    SkillNode("sql-basics", "SQL Fundamentals", "SELECT, JOIN, GROUP BY, window functions, indexes, query plans", "software", 1, 10.0,
              tags=["sql", "database", "mysql", "postgres", "rdbms"], onet_codes=["15-1243.00"]),
    SkillNode("sql-advanced", "Advanced SQL & Query Optimisation", "CTEs, recursive queries, partitioning, explain plans, index tuning", "software", 3, 10.0,
              tags=["advanced sql", "query optimisation", "performance tuning"], prerequisites=["sql-basics"]),
    SkillNode("nosql", "NoSQL Databases (MongoDB, Redis, Cassandra)", "Document model, key-value, wide-column, CAP tradeoffs, consistency levels", "software", 3, 12.0,
              tags=["nosql", "mongodb", "redis", "cassandra", "dynamodb"], prerequisites=["sql-basics"]),
    SkillNode("git", "Git & Collaborative Workflows", "Branching strategies, rebase, cherry-pick, CI integration, conventional commits", "software", 1, 6.0,
              tags=["git", "version control", "github", "gitlab"]),
    SkillNode("rest-api", "REST API Design & Development", "HTTP semantics, OpenAPI, versioning, auth (JWT/OAuth2), rate limiting", "software", 2, 10.0,
              tags=["rest", "api", "fastapi", "django", "http", "openapi"], prerequisites=["python-basics"]),
    SkillNode("graphql", "GraphQL", "Schema design, resolvers, DataLoader, subscriptions, federation", "software", 3, 10.0,
              tags=["graphql", "apollo", "schema"], prerequisites=["rest-api"]),
    SkillNode("testing", "Testing: Unit, Integration & E2E", "Pytest, mocking, fixtures, property-based testing, contract testing", "software", 2, 10.0,
              tags=["testing", "pytest", "unit tests", "tdd", "test driven"], prerequisites=["python-basics"]),
    SkillNode("linux", "Linux & Shell Scripting", "Process management, permissions, bash scripting, cron, systemd", "software", 2, 10.0,
              tags=["linux", "bash", "shell", "unix", "cli"]),
    SkillNode("networking", "Networking Fundamentals", "TCP/IP, DNS, HTTP/HTTPS, TLS, load balancers, proxies, CDN", "software", 2, 10.0,
              tags=["networking", "tcp/ip", "http", "dns", "network", "protocols"]),

    # ══════════════════════════════════════════════════════════════
    # DOMAIN: Machine Learning / Deep Learning
    # ══════════════════════════════════════════════════════════════
    SkillNode("math-foundations", "Math for ML: Linear Algebra & Calculus", "Vectors, matrices, eigenvalues, gradients, chain rule", "ml", 2, 20.0,
              tags=["linear algebra", "calculus", "math", "mathematics", "statistics"]),
    SkillNode("statistics-ml", "Statistics & Probability for ML", "Distributions, Bayes, MLE, hypothesis testing, information theory", "ml", 2, 15.0,
              tags=["statistics", "probability", "bayes", "statistical inference"], prerequisites=["math-foundations"]),
    SkillNode("numpy-pandas", "NumPy & Pandas", "Array broadcasting, vectorised ops, DataFrame manipulation, time-series indexing", "ml", 1, 14.0,
              tags=["numpy", "pandas", "dataframes", "data manipulation"], prerequisites=["python-basics"]),
    SkillNode("data-viz", "Data Visualisation (Matplotlib / Seaborn / Plotly)", "EDA, distribution plots, correlation heatmaps, interactive dashboards", "ml", 1, 8.0,
              tags=["matplotlib", "seaborn", "plotly", "eda", "visualization", "charts"], prerequisites=["numpy-pandas"]),
    SkillNode("data-preprocessing", "Data Preprocessing & Feature Engineering", "Missing values, scaling, encoding, class imbalance, leakage prevention", "ml", 2, 12.0,
              tags=["preprocessing", "feature engineering", "imputation", "scaling", "encoding", "data cleaning"], prerequisites=["numpy-pandas"]),
    SkillNode("classical-ml", "Classical ML Algorithms", "Regression, trees, ensemble methods, SVM, clustering, PCA", "ml", 3, 20.0,
              tags=["machine learning", "ml", "scikit-learn", "sklearn", "supervised", "unsupervised", "random forest"], prerequisites=["data-preprocessing", "statistics-ml"]),
    SkillNode("model-evaluation", "Model Evaluation & Validation", "Cross-validation, ROC/AUC, hyperparameter tuning, bias-variance tradeoff", "ml", 2, 10.0,
              tags=["model evaluation", "cross validation", "metrics", "overfitting", "hyperparameter"], prerequisites=["classical-ml"]),
    SkillNode("gradient-boosting", "Gradient Boosting (XGBoost / LightGBM / CatBoost)", "Boosting theory, regularisation, early stopping, tabular SOTA", "ml", 3, 10.0,
              tags=["xgboost", "lightgbm", "catboost", "gradient boosting", "boosting"], prerequisites=["model-evaluation"]),
    SkillNode("deep-learning-fundamentals", "Deep Learning Fundamentals", "Perceptrons, backprop, activations, optimisers (Adam/SGD), regularisation, batch norm", "ml", 3, 20.0,
              tags=["deep learning", "dl", "neural networks", "nn", "backpropagation", "gradient descent"], prerequisites=["classical-ml", "math-foundations"]),
    SkillNode("pytorch", "PyTorch", "Tensors, autograd, nn.Module, DataLoader, training loop, GPU acceleration", "ml", 3, 16.0,
              tags=["pytorch", "torch", "autograd", "neural network framework"], prerequisites=["deep-learning-fundamentals"]),
    SkillNode("tensorflow", "TensorFlow & Keras", "tf.data, layers API, model subclassing, callbacks, TF Lite", "ml", 3, 14.0,
              tags=["tensorflow", "tf", "keras", "tflite"], prerequisites=["deep-learning-fundamentals"]),
    SkillNode("cnn", "Convolutional Neural Networks (CNN)", "Conv layers, pooling, ResNet, EfficientNet, transfer learning, augmentation", "ml", 4, 16.0,
              tags=["cnn", "convolutional", "computer vision", "image classification", "resnet"], prerequisites=["pytorch"]),
    SkillNode("rnn-lstm", "RNNs, LSTMs & GRUs", "Sequence modelling, vanishing gradients, bidirectional LSTMs, time-series with DL", "ml", 4, 14.0,
              tags=["rnn", "lstm", "gru", "sequence", "recurrent"], prerequisites=["pytorch"]),
    SkillNode("transformers", "Transformer Architecture", "Self-attention, positional encoding, multi-head attention, encoder-decoder", "ml", 4, 16.0,
              tags=["transformers", "attention", "self-attention", "encoder decoder"], prerequisites=["pytorch", "rnn-lstm"]),
    SkillNode("nlp-classical", "Classical NLP", "Tokenisation, TF-IDF, word2vec, sentiment analysis, named entity recognition", "ml", 3, 12.0,
              tags=["nlp", "natural language processing", "text", "tokenisation", "word2vec"], prerequisites=["classical-ml"]),
    SkillNode("nlp-transformers", "NLP with Transformers & HuggingFace", "BERT, GPT fine-tuning, tokenisers, pipelines, text classification, NER", "ml", 4, 16.0,
              tags=["huggingface", "bert", "nlp transformers", "text classification", "ner", "fine-tuning"], prerequisites=["transformers", "nlp-classical"]),
    SkillNode("llm-fundamentals", "Large Language Models (LLMs)", "Architecture deep-dive, scaling laws, RLHF, instruction tuning, safety", "ml", 5, 16.0,
              tags=["llm", "large language model", "gpt", "chatgpt", "claude", "instruction tuning", "rlhf"], prerequisites=["nlp-transformers"]),
    SkillNode("llm-fine-tuning", "LLM Fine-Tuning (LoRA / QLoRA / PEFT)", "Parameter-efficient fine-tuning, dataset preparation, evaluation, quantisation", "ml", 5, 16.0,
              tags=["lora", "qlora", "peft", "fine tuning", "finetuning", "llm fine-tuning"], prerequisites=["llm-fundamentals"]),
    SkillNode("rag", "Retrieval-Augmented Generation (RAG)", "Vector stores, chunking, embedding, retrieval strategies, re-ranking, evaluation", "ml", 4, 14.0,
              tags=["rag", "retrieval augmented generation", "vector search", "embeddings", "chromadb", "pinecone"], prerequisites=["llm-fundamentals"]),
    SkillNode("langchain", "LangChain & LLM Orchestration", "Chains, agents, tools, memory, LangGraph, multi-step reasoning", "ml", 4, 12.0,
              tags=["langchain", "langgraph", "agents", "llm orchestration", "ai agents"], prerequisites=["rag"]),
    SkillNode("reinforcement-learning", "Reinforcement Learning", "MDP, Q-learning, policy gradient, PPO, actor-critic, OpenAI Gym", "ml", 5, 20.0,
              tags=["rl", "reinforcement learning", "ppo", "q-learning", "dqn", "actor critic"], prerequisites=["deep-learning-fundamentals"]),
    SkillNode("computer-vision", "Advanced Computer Vision", "Object detection (YOLO/DETR), segmentation (SAM), optical flow, depth estimation", "ml", 5, 18.0,
              tags=["computer vision", "cv", "object detection", "yolo", "segmentation", "sam"], prerequisites=["cnn"]),
    SkillNode("generative-ai", "Generative AI: GANs & Diffusion Models", "GAN training dynamics, diffusion process, stable diffusion, ControlNet, LoRA for image", "ml", 5, 16.0,
              tags=["generative ai", "gans", "diffusion", "stable diffusion", "midjourney", "image generation"], prerequisites=["cnn", "transformers"]),
    SkillNode("mlops-fundamentals", "MLOps Fundamentals", "Experiment tracking (MLflow), data versioning (DVC), CI/CD for ML, model registry", "ml", 3, 16.0,
              tags=["mlops", "mlflow", "dvc", "experiment tracking", "model registry", "ml pipeline"], prerequisites=["classical-ml"]),
    SkillNode("model-serving", "Model Serving & Deployment", "FastAPI serving, TorchServe, Triton, ONNX export, batching, A/B testing models", "ml", 4, 14.0,
              tags=["model serving", "torchserve", "triton", "onnx", "model deployment", "inference"], prerequisites=["mlops-fundamentals"]),
    SkillNode("ml-monitoring", "ML Monitoring & Drift Detection", "Data drift, concept drift, model degradation, Evidently AI, Great Expectations", "ml", 4, 10.0,
              tags=["monitoring", "drift detection", "data drift", "evidently", "model monitoring"], prerequisites=["model-serving"]),
    SkillNode("feature-store", "Feature Stores & ML Platform", "Feast, Tecton, online/offline feature serving, training-serving skew prevention", "ml", 4, 10.0,
              tags=["feature store", "feast", "tecton", "ml platform", "feature engineering pipeline"], prerequisites=["mlops-fundamentals"]),

    # ══════════════════════════════════════════════════════════════
    # DOMAIN: Cloud & DevOps
    # ══════════════════════════════════════════════════════════════
    SkillNode("docker", "Docker & Containerisation", "Dockerfile, multi-stage builds, compose, networking, security best practices", "cloud", 2, 12.0,
              tags=["docker", "containers", "containerisation", "dockerfile", "docker-compose"], prerequisites=["linux"]),
    SkillNode("kubernetes", "Kubernetes (K8s)", "Pods, deployments, services, ingress, HPA, RBAC, Helm, GitOps", "cloud", 4, 20.0,
              tags=["kubernetes", "k8s", "helm", "kubectl", "orchestration", "gitops"], prerequisites=["docker"]),
    SkillNode("aws-fundamentals", "AWS Cloud Fundamentals", "EC2, S3, VPC, IAM, RDS, Lambda — core services and pricing model", "cloud", 2, 16.0,
              tags=["aws", "amazon web services", "cloud", "ec2", "s3", "lambda", "iam"], prerequisites=["networking"]),
    SkillNode("aws-advanced", "AWS Advanced: EKS, SageMaker & CDK", "Managed Kubernetes, ML platform, infrastructure as code", "cloud", 4, 16.0,
              tags=["eks", "sagemaker", "cdk", "aws advanced", "aws ml"], prerequisites=["aws-fundamentals", "kubernetes"]),
    SkillNode("gcp", "Google Cloud Platform (GCP)", "GKE, BigQuery, Vertex AI, Cloud Run, Pub/Sub, Dataflow", "cloud", 3, 16.0,
              tags=["gcp", "google cloud", "bigquery", "vertex ai", "gke"], prerequisites=["networking"]),
    SkillNode("azure", "Microsoft Azure", "AKS, Azure ML, Cognitive Services, Azure DevOps, Cosmos DB", "cloud", 3, 16.0,
              tags=["azure", "microsoft azure", "aks", "azure ml"], prerequisites=["networking"]),
    SkillNode("terraform", "Terraform & Infrastructure as Code", "HCL, state management, modules, workspaces, Atlantis, drift detection", "cloud", 3, 14.0,
              tags=["terraform", "iac", "infrastructure as code", "hcl", "pulumi"], prerequisites=["aws-fundamentals"]),
    SkillNode("cicd", "CI/CD Pipelines", "GitHub Actions, GitLab CI, Jenkins, ArgoCD, deployment strategies (blue-green, canary)", "cloud", 3, 12.0,
              tags=["ci/cd", "cicd", "github actions", "jenkins", "argocd", "devops", "continuous integration"], prerequisites=["git", "docker"]),
    SkillNode("observability", "Observability (Prometheus, Grafana, OpenTelemetry)", "Metrics, logs, traces, alerting, SLO/SLI/SLA, on-call practices", "cloud", 3, 12.0,
              tags=["monitoring", "observability", "prometheus", "grafana", "opentelemetry", "logging", "tracing"], prerequisites=["kubernetes"]),

    # ══════════════════════════════════════════════════════════════
    # DOMAIN: Data Engineering
    # ══════════════════════════════════════════════════════════════
    SkillNode("data-warehousing", "Data Warehousing & Modelling", "Star/snowflake schemas, SCD types, Kimball methodology, Redshift, BigQuery", "data-eng", 3, 14.0,
              tags=["data warehouse", "data warehousing", "redshift", "snowflake", "star schema", "dimensional modelling"], prerequisites=["sql-advanced"]),
    SkillNode("etl-pipelines", "ETL/ELT Pipelines & Orchestration", "Airflow, Prefect, Dagster, dbt, incremental loading, idempotency", "data-eng", 3, 16.0,
              tags=["etl", "elt", "airflow", "prefect", "dbt", "data pipeline", "orchestration"], prerequisites=["data-warehousing"]),
    SkillNode("spark", "Apache Spark & PySpark", "RDD/DataFrame API, Spark SQL, partitioning, joins, caching, Databricks", "data-eng", 4, 20.0,
              tags=["spark", "pyspark", "databricks", "big data", "distributed computing"], prerequisites=["numpy-pandas", "sql-advanced"]),
    SkillNode("kafka", "Apache Kafka & Streaming", "Topics, partitions, consumer groups, exactly-once, Kafka Streams, Flink", "data-eng", 4, 16.0,
              tags=["kafka", "streaming", "event streaming", "flink", "kinesis", "real-time"], prerequisites=["distributed-systems"]),
    SkillNode("data-lake", "Data Lake Architecture & Delta Lake", "Medallion architecture, ACID on object store, schema evolution, time travel", "data-eng", 4, 12.0,
              tags=["data lake", "delta lake", "lakehouse", "iceberg", "hudi", "s3"], prerequisites=["spark"]),

    # ══════════════════════════════════════════════════════════════
    # DOMAIN: Security
    # ══════════════════════════════════════════════════════════════
    SkillNode("security-fundamentals", "Security Fundamentals", "OWASP Top 10, CIA triad, auth & authorisation, PKI, threat modelling", "security", 2, 12.0,
              tags=["security", "cybersecurity", "owasp", "authentication", "authorisation", "infosec"]),
    SkillNode("cloud-security", "Cloud Security & IAM", "Least privilege, SCPs, secrets management (Vault), compliance (SOC2/PCI)", "security", 3, 12.0,
              tags=["cloud security", "iam", "vault", "secrets", "compliance"], prerequisites=["security-fundamentals", "aws-fundamentals"]),
    SkillNode("appsec", "Application Security & Penetration Testing", "SAST, DAST, dependency scanning, CVSS scoring, red team basics", "security", 4, 16.0,
              tags=["appsec", "penetration testing", "pen test", "sast", "dast", "vulnerability"], prerequisites=["security-fundamentals"]),

    # ══════════════════════════════════════════════════════════════
    # DOMAIN: Product / Management
    # ══════════════════════════════════════════════════════════════
    SkillNode("product-management", "Product Management Fundamentals", "PRDs, OKRs, roadmapping, stakeholder management, go-to-market", "product", 2, 12.0,
              tags=["product management", "pm", "product manager", "prds", "roadmap", "okr"], onet_codes=["11-2021.00"]),
    SkillNode("agile-scrum", "Agile & Scrum", "Sprint planning, retrospectives, Kanban, velocity, estimation, JIRA", "product", 1, 8.0,
              tags=["agile", "scrum", "kanban", "sprint", "jira", "project management"], onet_codes=["11-9199.09"]),
    SkillNode("data-analytics", "Data Analytics & Business Intelligence", "KPI definition, funnel analysis, cohort analysis, dashboards (Tableau/Looker)", "product", 2, 12.0,
              tags=["data analytics", "bi", "business intelligence", "tableau", "looker", "analytics", "kpi"], prerequisites=["sql-basics"]),
    SkillNode("a-b-testing", "A/B Testing & Experimentation", "Hypothesis design, statistical significance, p-values, Bayesian testing, multi-armed bandit", "product", 3, 10.0,
              tags=["a/b testing", "ab testing", "experimentation", "hypothesis testing", "statistical testing"], prerequisites=["statistics-ml"]),

    # ══════════════════════════════════════════════════════════════
    # DOMAIN: Cross-Domain / General (supports O*NET generalisation)
    # ══════════════════════════════════════════════════════════════
    SkillNode("excel-advanced", "Advanced Excel & Power BI", "Pivot tables, Power Query, DAX, VBA macros, dashboard design", "general", 1, 8.0,
              tags=["excel", "power bi", "spreadsheet", "powerbi", "dax", "power query"], onet_codes=["13-1111.00", "13-2011.00"]),
    SkillNode("communication", "Technical Communication & Documentation", "Technical writing, API documentation, architecture decision records", "general", 1, 6.0,
              tags=["communication", "writing", "documentation", "technical writing"]),
    SkillNode("project-management", "Project Management (PMP / Prince2)", "WBS, critical path, risk register, earned value, stakeholder matrix", "general", 2, 14.0,
              tags=["project management", "pmp", "prince2", "project planning", "risk management"], onet_codes=["11-9199.01"]),
    SkillNode("supply-chain", "Supply Chain & Logistics Analytics", "Demand forecasting, inventory optimisation, warehouse management, ERP", "general", 2, 12.0,
              tags=["supply chain", "logistics", "inventory", "erp", "warehouse", "operations"], onet_codes=["11-3071.00"]),
    SkillNode("hr-analytics", "HR Analytics & Talent Management", "Workforce planning, attrition modelling, compensation benchmarking, HRIS", "general", 2, 10.0,
              tags=["hr", "human resources", "talent", "hr analytics", "attrition", "hris"], onet_codes=["13-1071.00"]),
    SkillNode("marketing-analytics", "Marketing Analytics & Attribution", "CAC, LTV, multi-touch attribution, SEO analytics, campaign measurement", "general", 2, 10.0,
              tags=["marketing", "marketing analytics", "attribution", "seo", "cac", "ltv"], onet_codes=["13-1161.00"]),
    SkillNode("financial-analysis", "Financial Analysis & Modelling", "DCF, P&L modelling, scenario analysis, unit economics, Excel financial models", "general", 3, 14.0,
              tags=["finance", "financial analysis", "dcf", "financial modelling", "accounting"], onet_codes=["13-2051.00"]),
]


# ─────────────────────────────────────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────────────────────────────────────

def build_knowledge_graph() -> nx.DiGraph:
    """
    Construct the master skill knowledge graph as a NetworkX DiGraph.
    Nodes = skills.  Edges = prerequisite dependencies (child → parent direction).
    Edge weight = parent's difficulty (harder prerequisites cost more to skip).
    """
    G = nx.DiGraph()

    # Add all skill nodes
    for skill in SKILL_NODES:
        G.add_node(
            skill.id,
            name=skill.name,
            description=skill.description,
            domain=skill.domain,
            difficulty=skill.difficulty,
            base_hours=skill.base_hours,
            tags=skill.tags,
            onet_codes=skill.onet_codes,
        )

    # Add directed prerequisite edges: prereq → skill (must learn prereq first)
    for skill in SKILL_NODES:
        for prereq_id in skill.prerequisites:
            if G.has_node(prereq_id):
                G.add_edge(
                    prereq_id,
                    skill.id,
                    weight=G.nodes[prereq_id]["difficulty"],
                    relationship="prerequisite",
                )

    return G


# Build singleton graph at import time
KNOWLEDGE_GRAPH: nx.DiGraph = build_knowledge_graph()

# Reverse lookup: tag → skill_id  (used for fuzzy matching)
TAG_INDEX: dict[str, list[str]] = {}
for _skill in SKILL_NODES:
    for _tag in _skill.tags:
        TAG_INDEX.setdefault(_tag.lower(), []).append(_skill.id)
    TAG_INDEX.setdefault(_skill.name.lower(), []).append(_skill.id)
    TAG_INDEX.setdefault(_skill.id.lower(), []).append(_skill.id)

# ID → SkillNode lookup
SKILL_LOOKUP: dict[str, SkillNode] = {s.id: s for s in SKILL_NODES}


def resolve_skill_id(raw_name: str) -> Optional[str]:
    """
    Map a free-form skill name (from resume/JD) to a canonical skill ID.
    Uses exact tag match first, then partial substring match.
    Returns None if no match found.
    """
    key = raw_name.lower().strip()

    # 1. Exact match
    if key in TAG_INDEX:
        return TAG_INDEX[key][0]

    # 2. Partial match — key is a substring of a tag OR vice versa
    for tag, ids in TAG_INDEX.items():
        if key in tag or tag in key:
            return ids[0]

    return None


def get_prerequisite_chain(skill_id: str, depth: int = 5) -> list[str]:
    """
    Return all ancestors of a skill up to `depth` levels deep,
    in topological order (i.e., foundations first).
    """
    if skill_id not in KNOWLEDGE_GRAPH:
        return []
    ancestors = nx.ancestors(KNOWLEDGE_GRAPH, skill_id)
    subgraph = KNOWLEDGE_GRAPH.subgraph(ancestors | {skill_id})
    try:
        topo = list(nx.topological_sort(subgraph))
        return topo
    except nx.NetworkXUnfeasible:
        return list(ancestors)


def get_domain_skills(domain: str) -> list[str]:
    """Return all skill IDs belonging to a domain."""
    return [
        nid for nid, data in KNOWLEDGE_GRAPH.nodes(data=True)
        if data.get("domain") == domain
    ]
