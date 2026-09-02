# VYORIS Google Cloud Run Deployment Guide

This document outlines the end-to-end sequential steps we took to authenticate, configure, and successfully deploy the VYORIS application to Google Cloud Run. It also provides instructions for verifying the deployment and setting up GitHub-based Continuous Deployment (CI/CD).

---

## Part 1: Local Authentication (User Steps)

Before deploying, the local `gcloud` CLI was configured to point to the correct Google Cloud project and authorized to act on your behalf.

1. **Set Active Project:**  
   You attempted to set the project ID and eventually resolved permissions to authenticate against `vyoris-507407`.
   ```bash
   gcloud config set project vyoris-507407
   ```

2. **Set Application Default Credentials (ADC):**  
   You updated the Application Default Credentials to ensure local code and the `gcloud` CLI billed quotas to the correct project.
   ```bash
   gcloud auth application-default set-quota-project vyoris-507407
   ```

---

## Part 2: Application Configuration (Agent Steps)

To prepare the FastAPI application for Cloud Run's containerized environment, we modified the source code and generated essential Docker files.

1. **Update FastAPI Port Binding:**  
   Modified `src/agents/agent_orchestration.py` to dynamically bind to the `$PORT` environment variable provided by Cloud Run, rather than a hardcoded port.
   ```python
   port = int(os.environ.get("PORT", 8080))
   uvicorn.run("src.agents.agent_orchestration:app", host="0.0.0.0", port=port, reload=True)
   ```

2. **Create `.dockerignore`:**  
   Created a `.dockerignore` file to strictly prevent sensitive files (like `.env`) and bloated directories (like `.venv` and `__pycache__`) from being uploaded to Cloud Build or baked into the container.

3. **Create `Dockerfile`:**  
   Generated a production-ready `Dockerfile` using a lightweight Python base image (`python:3.10-slim`). The Dockerfile installs dependencies via `requirements.txt`, copies the source code, and sets the `CMD` to start `uvicorn`.

---

## Part 3: Deployment and Troubleshooting (Agent Steps)

During the deployment process, we encountered a few standard GCP hurdles and resolved them iteratively.

1. **Initial Deployment Attempt:**  
   Executed the deploy command from source.
   ```bash
   gcloud run deploy vyoris-backend --source . --region asia-south1 --allow-unauthenticated --project vyoris-507407
   ```
   *Issue:* The build failed with an `INVALID_ARGUMENT: ... does not have storage.objects.get access` error.

2. **Fixing IAM Permissions (Cloud Build):**  
   The default Compute Engine service account used by Cloud Build lacked permissions to read the source zip, write the logs, and push the final image to Artifact Registry. I ran the following commands to grant the necessary roles:
   ```bash
   gcloud projects add-iam-policy-binding vyoris-507407 \
       --member="serviceAccount:1052299082607-compute@developer.gserviceaccount.com" \
       --role="roles/storage.objectViewer"
       
   gcloud projects add-iam-policy-binding vyoris-507407 \
       --member="serviceAccount:1052299082607-compute@developer.gserviceaccount.com" \
       --role="roles/logging.logWriter"
       
   gcloud projects add-iam-policy-binding vyoris-507407 \
       --member="serviceAccount:1052299082607-compute@developer.gserviceaccount.com" \
       --role="roles/artifactregistry.writer"
   ```

3. **Resolving Out-Of-Memory (OOM) Errors:**  
   After fixing the permissions, the image built successfully but the container crashed on startup. Logs revealed: `Memory limit of 512 MiB exceeded`. Because the app loads heavy Machine Learning libraries (`torch`, `transformers`), the default 512MB limit was instantly exceeded.

4. **Final Successful Deployment:**  
   Redeployed the application, explicitly allocating **4Gi (4 Gigabytes)** of memory to the container:
   ```bash
   gcloud run deploy vyoris-backend --source . --region asia-south1 --allow-unauthenticated --project vyoris-507407 --memory 4Gi
   ```
   *Result:* The deployment succeeded and the app is now serving traffic at:  
   **https://vyoris-backend-1052299082607.asia-south1.run.app**

---

## Part 4: How to Verify the Deployment in the GCP Portal

To check on your application's health, view logs, or adjust settings via the web interface:

1. **Go to Cloud Run:**  
   Open the Google Cloud Console and navigate to **Cloud Run** (or search for "Cloud Run" in the top search bar).
2. **Select your Service:**  
   Click on the **`vyoris-backend`** service.
3. **Explore the Dashboard:**
   - **Metrics Tab:** View traffic volume, request latency, and memory utilization to ensure the 4Gi limit is sufficient.
   - **Logs Tab:** View real-time application logs. This is exactly where you will see any Python `print()` or `logger.info()` outputs from your agent orchestration code.
   - **Revisions Tab:** See the history of all your deployments. You can roll back to a previous version instantly from here.

---

## Part 5: Manually Verifying Logs via CLI

If you prefer to check logs directly from your local terminal instead of using the GCP Web Portal, you can use the `gcloud` CLI. This is incredibly helpful for quickly debugging startup issues or monitoring incoming API requests.

**1. Tail Live Logs:**
To stream the live application logs directly in your terminal, run:
```bash
gcloud beta run services logs tail vyoris-backend --region asia-south1 --project vyoris-507407
```
*(Press `Ctrl+C` to stop streaming)*

**2. Read Recent Error Logs:**
If the application crashes, you can fetch recent errors using advanced filtering:
```bash
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=vyoris-backend AND severity>=WARNING" --project=vyoris-507407 --limit=20
```

---

## Part 6: Linking your GitHub Repo (`vyoris_test`) for Continuous Deployment (CI/CD)

Right now, you deployed the application manually from your local machine using `--source .`. To automate this so that every `git push` to your GitHub repository automatically deploys to Cloud Run, follow these steps:

1. **Go to Cloud Run:** Open the `vyoris-backend` service in the Google Cloud Console.
2. **Setup Continuous Deployment:** At the top of the service details page, click the button labeled **"Set up Continuous Deployment"**.
3. **Authenticate GitHub:** Select **GitHub** as the provider and click to authenticate your GitHub account. 
4. **Select Repository:** Choose your `vyoris_test` repository from the dropdown list.
5. **Configure the Build:**
   - **Branch:** Select the branch you want to deploy from (e.g., `^main$` or `^master$`).
   - **Build Type:** Select **Dockerfile**.
   - **Source Location:** Leave it as `/Dockerfile` (since it's in the root of your repo).
6. **Save and Trigger:** Click **Save**. 

Google Cloud will automatically create a **Cloud Build Trigger**. From now on, whenever you push code to the selected branch on GitHub, Cloud Build will automatically build a new container image and deploy it to Cloud Run.
