# Venkata Raghunadh Illinda

Hyderabad, India | +91 93917 24488 | [PERSONAL EMAIL - REPLACE] | linkedin.com/in/askillinda | github.com/illindva72

## Professional Summary

MLOps / AIOps Engineer with 18 years in production application support and reliability for investment banking (prime brokerage: securities lending, margin financing, trade execution, risk reporting). Applies SRE discipline (SLOs/SLIs, observability, incident automation) to ML and AI systems. Builds and operates Python ML services (FastAPI, Flask, Streamlit), containerised with Docker, deployed through CI/CD to Azure, and exposed to LLM agents via Model Context Protocol (MCP). Completing an MSc in AI & ML (IIIT Bangalore and Liverpool John Moores University, via upGrad); final project and thesis due December 2026.

## Core Skills

- **MLOps:** ML pipelines, model packaging and serving, model evaluation, dataset/leakage and drift-scenario testing, model maintenance via pipelines
- **AIOps / Observability:** AppDynamics, Splunk, Grafana, Amelia (AIOps platform), alerting, telemetry, SLO/SLI design, incident automation
- **Languages:** Python, Bash, SQL (PostgreSQL, Oracle PL/SQL, MSSQL, Sybase); basic Java/C# debugging
- **ML / AI:** EDA, data engineering, PyTorch, PyTorch Forecasting (Temporal Fusion Transformer), HuggingFace Transformers (FinBERT), LangChain/LangGraph, LLM prompt engineering
- **Serving / Web:** FastAPI, Flask, Streamlit, Django, HTMX
- **Agentic / MCP:** MCP server design and tool exposure, LLM tool-calling agents
- **Cloud / DevOps:** Azure (AKS, App Service, storage), Docker, Kubernetes, GitHub Actions, GitLab CI/CD, Snowflake, ETL and batch/streaming pipelines
- **Practices:** ITIL, incident and problem management, disaster-recovery planning, dashboards and operational reporting

## Professional Experience

### Associate Director | UBS Business Solutions, Hyderabad
May 2025 - Present (UBS Business Solutions LLC, Mar 2023 - Apr 2025)

- Led technical teams supporting enterprise-scale prime brokerage applications in investment banking
- Implemented SRE practices across cloud and on-premises applications; defined and monitored SLOs and SLIs for availability and performance
- Deployed observability, alerting and telemetry using AppDynamics and Splunk; leading the migration to Grafana and Amelia (AIOps)
- Automated incident-management workflows to cut response and resolution times on critical issues
- Maintained CI/CD pipelines following DevOps best practice for secure, repeatable deployments
- Operated Kubernetes and AKS environments for containerised, cloud-native applications
- Architected and supported data platforms on Azure storage and Snowflake: ETL pipelines, batch jobs and streaming services
- Built dashboards and reports giving stakeholders a clear view of operational health
- Mentored engineers on SRE principles, observability and cloud application strategy; led vendor evaluations

### Authorized Officer | UBS Business Solutions LLC
Oct 2018 - Mar 2023

- Monitored system health and performance of trading and prime brokerage platforms, resolving issues before business impact
- Wrote and maintained automation scripts for routine operational tasks, reducing manual effort and human error
- Implemented incident-management processes that shortened response time to critical outages
- Built real-time dashboards and reporting for trading staff and management
- Contributed to disaster-recovery planning and testing for business continuity
- Acted as primary contact for major upgrades, coordinating development, QA and operations
- Audited system access and security controls with the cybersecurity team

### Earlier Roles

- **Software Development Advisor**, NTT Data (formerly Dell Services), May 2016 - Sep 2018
- **Senior Software Engineer**, NTT Data (formerly Dell Services), May 2014 - May 2016
- **Senior Application Developer**, APAR Technologies, Sep 2013 - Mar 2014
- **Senior Software Analyst**, Dell Perot Systems and Cognizant, Jun 2009 - Sep 2013
- **Software Engineer**, CES India Pvt. Ltd., Sep 2008 - Jun 2009

## Selected AI/ML Projects

### VYORIS: Quantitative Market Analysis Platform (MSc project)
Python, FastAPI, MCP, LangChain/LangGraph, Anthropic Claude, PyTorch Forecasting (TFT), HuggingFace FinBERT, Supabase/PostgreSQL, HTMX, Azure App Service, GitHub Actions

- Built an agentic web platform for NSE/BSE stocks: a FastAPI service with a LangGraph ReAct agent (Claude) that calls four tools (market data, news sentiment, forecast, model metrics) and returns a two-audience briefing (retail and quant)
- Built an MCP server that exposes the same four tools to any MCP client over the Model Context Protocol; the web agent reuses the tool functions in-process
- Wrote the agent system prompt with a structured prompt framework (context, objective, ordered steps, audience, strict output format)
- Implemented FinBERT news-sentiment scoring (confidence-weighted, -1 to +1) and data ingestion with yfinance, Z-score outlier clipping and scaling
- Wrote TFT and LSTM-baseline training and evaluation code (RMSE, MAE, MAPE, R2, attention-weight extraction); the served forecast and metrics tools currently return placeholder values until a trained checkpoint is wired in
- Built Supabase email-OTP authentication, per-user search history with retention limits, and a background NSE symbol sync (batched upserts, 24-hour freshness check, trigram search)
- Added a pytest suite for the data pipeline, MCP tools, agent orchestration, latency and a market-crash scenario (March 2020); automated deploy to Azure App Service with GitHub Actions
- Repo: github.com/vrilinda/vyoris_test

### [SECOND PROJECT - ASKI RESEARCH LABS: DETAILS PENDING]
- To be added once the repository contents are confirmed (see notes)

## Education

- **MSc, Artificial Intelligence & Machine Learning**: IIIT Bangalore (Year 1) and Liverpool John Moores University (Year 2), via upGrad. In progress; thesis and final project due Dec 2026; degree certificate expected Feb 2027
- **BSc, Mathematics**: Andhra University, Visakhapatnam, 2002 - 2005
