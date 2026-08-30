import os
import sys
import logging
from fastapi.testclient import TestClient

# Add project root to sys.path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.agent_orchestration import app

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("test_gui")

client = TestClient(app)

def test_gui_homepage():
    """
    Test Phase 1: Verify the Jinja2 index.html is served correctly.
    """
    logger.info("--- Phase 1: Testing Main Dashboard Endpoint (GET /) ---")
    response = client.get("/")
    
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"
    assert "VYORIS Financial Dashboard" in response.text, "Dashboard title missing in HTML"
    assert "hx-post=\"/htmx/analyze\"" in response.text, "HTMX form missing in HTML"
    logger.info("SUCCESS: Homepage served correctly with HTMX integrations.")

def test_gui_htmx_analyze():
    """
    Test Phase 2: Verify the HTMX POST endpoint successfully invokes the LangChain Agent 
    and returns the insight_partial.html fragment.
    """
    logger.info("--- Phase 2: Testing HTMX Analysis Endpoint (POST /htmx/analyze) ---")
    
    # Send a form-data request simulating the HTMX form submission
    response = client.post(
        "/htmx/analyze",
        data={"ticker": "RELIANCE.NS"}
    )
    
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"
    
    # Verify the partial HTML is returned (not the full index.html)
    assert "VYORIS Financial Dashboard" not in response.text, "Expected a partial HTML fragment, but got the full layout."
    
    # Verify the insight was generated and injected
    assert "Analysis:" in response.text, "Expected 'Analysis:' header in partial HTML."
    assert "RELIANCE.NS" in response.text, "Expected ticker name in partial HTML."
    
    logger.info("SUCCESS: HTMX endpoint processed ticker and returned valid HTML fragment.")

if __name__ == "__main__":
    logger.info("Starting End-User GUI Tests...\n")
    try:
        test_gui_homepage()
        print("\n")
        test_gui_htmx_analyze()
        print("\n")
        logger.info("ALL GUI TESTS PASSED SUCCESSFULLY! ✅")
    except AssertionError as e:
        logger.error(f"TEST FAILED ❌: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"UNEXPECTED ERROR ❌: {e}")
        sys.exit(1)
