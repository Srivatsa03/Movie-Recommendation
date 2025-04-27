from fastapi import FastAPI, Path
from app.models import SVDRecommender
from app.logger import logger
from app.schema import RecommendationsResponse
from datetime import datetime
from app.provenance import log_provenance 
import ldclient
from ldclient.config import Config
from ldclient.context import Context
import os
from dotenv import load_dotenv

load_dotenv()
ld_sdk_key = os.getenv("LD_SDK_KEY")  
ldclient.set_config(Config(sdk_key=ld_sdk_key))
ld_client = ldclient.get()

logger.info("Initializing Model A (Trained Model)")
model_a = SVDRecommender()
logger.info("Finished initializing Model A")
#svd = SVDRecommender()

logger.info("Initializing Model B (Retrained Model)")
model_b = SVDRecommender(model_path="retrained_models/svdpp_model_20250425_095111.pkl")

logger.info("Finished initializing Model B")

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
    user_context = Context.builder(str(user_id)).kind("user").build()

    model_assignment = ld_client.variation("ab-testing", user_context, "model_a")
    logger.info(f"LaunchDarkly assigned model: {model_assignment}")


    if model_assignment == "model_a":
        selected_model = model_a
    else:
        selected_model = model_b

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

    # Also log to provenance file
    from app.provenance import log_provenance  # Make sure this import is present
    log_provenance(
        user_id=user_id,
        model_version=str(selected_model.saved_model_path),
        data_version=str(selected_model.dataset_path),
        recommendations=recommendations
    )

    return RecommendationsResponse(recommendations=recommendations)
