import os
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, Path

from app.logger import logger
from app.models import SVDRecommender
from app.provenance import log_provenance
from app.schema import RecommendationsResponse

try:
    import ldclient
    from ldclient.config import Config
    from ldclient.context import Context
except ModuleNotFoundError:  # pragma: no cover - fallback for local/test runs
    ldclient = None
    Config = None
    Context = None

load_dotenv()
ld_sdk_key = os.getenv("LD_SDK_KEY")


class _FallbackLDClient:
    def variation(self, _flag_key, _context, default):
        return default


if ldclient and ld_sdk_key:
    ldclient.set_config(Config(sdk_key=ld_sdk_key))
    ld_client = ldclient.get()
else:
    logger.warning(
        "LaunchDarkly SDK unavailable or LD_SDK_KEY not set; defaulting to model_a."
    )
    ld_client = _FallbackLDClient()

logger.info("Initializing Model A (Trained Model)")
model_a = SVDRecommender()
logger.info("Finished initializing Model A")

logger.info("Initializing Model B (Retrained Model - 4/25)")
model_b = SVDRecommender(model_path="retrained_models/svdpp_model_20250425_095111.pkl")
logger.info("Finished initializing Model B")

logger.info("Initializing Model C (Retrained Model - 4/21)")
model_c = SVDRecommender(model_path="retrained_models/svdpp_model_20250421_170540.pkl")
logger.info("Finished initializing Model C")

logger.info("Initializing Model D (Retrained Model - 4/22)")
model_d = SVDRecommender(model_path="retrained_models/svdpp_model_20250422_160631.pkl")
logger.info("Finished initializing Model D")

logger.info("Initializing Model E (Retrained Model - 4/23)")
model_e = SVDRecommender(model_path="retrained_models/svdpp_model_20250423_191258.pkl")
logger.info("Finished initializing Model E")

app = FastAPI()


@app.get(
    "/recommend/{user_id}",
    response_model=RecommendationsResponse,
    responses={200: {"description": "Success"}},
)
async def get_recommendations(
    user_id: int = Path(..., description="User ID to get recommendations for"),
):
    """
    Generates recommendations for a given user and logs provenance information
    """
    if Context:
        user_context = Context.builder(str(user_id)).kind("user").build()
    else:
        user_context = {"key": str(user_id), "kind": "user"}

    model_assignment = ld_client.variation("ab-testing", user_context, "model_a")
    logger.info(f"LaunchDarkly assigned model: {model_assignment}")

    selected_model = model_a

    if model_assignment == "model_b":
        selected_model = model_b
    elif model_assignment == "model_c":
        selected_model = model_c
    elif model_assignment == "model_d":
        selected_model = model_d
    elif model_assignment == "model_e":
        selected_model = model_e

    # Generate recommendations
    recommendations, _ = selected_model.recommend(user_id)

    # Convert recommendations to strings for the schema
    recommendations = [str(rec) for rec in recommendations]  # 🔥 This is essential

    # Provenance Logging
    provenance_info = {
        "timestamp": datetime.utcnow().isoformat(),
        "user_id": user_id,
        "model_version": str(selected_model.saved_model_path),
        "data_version": str(selected_model.dataset_path),
        "num_recommendations": len(recommendations),
    }
    logger.info(f"Provenance Log: {provenance_info}")

    log_provenance(
        user_id=user_id,
        model_version=str(selected_model.saved_model_path),
        data_version=str(selected_model.dataset_path),
        recommendations=recommendations,
    )

    return RecommendationsResponse(recommendations=recommendations)
