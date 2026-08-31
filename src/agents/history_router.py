from fastapi import APIRouter, Request, Depends, HTTPException, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from supabase import create_client, Client
import logging
from typing import List

from src.config import config
from src.agents.auth_router import require_auth, get_supabase, get_current_user

logger = logging.getLogger("HistoryRouter")
router = APIRouter()
templates = Jinja2Templates(directory="src/templates")

def enforce_history_retention(user_id: str, supabase: Client):
    """
    Retains all favorites, but strictly keeps only the top 10 most recent non-favorite searches.
    Deletes the rest.
    """
    try:
        res = supabase.table("search_history")\
            .select("id")\
            .eq("user_id", user_id)\
            .eq("is_favorite", False)\
            .order("created_at", desc=True)\
            .execute()
            
        records = res.data
        if len(records) > 10:
            ids_to_delete = [r["id"] for r in records[10:]]
            supabase.table("search_history").delete().in_("id", ids_to_delete).execute()
            logger.info(f"Deleted {len(ids_to_delete)} old search history records for user {user_id}")
    except Exception as e:
        logger.error(f"Error enforcing history retention: {e}")

@router.get("/history", response_class=HTMLResponse)
async def view_history(request: Request, user = Depends(require_auth)):
    """Renders the user's search history dashboard."""
    token = request.cookies.get("vyoris_access_token")
    supabase = get_supabase(token)
    try:
        # Fetch favorites first, then recents
        fav_res = supabase.table("search_history")\
            .select("*")\
            .eq("user_id", user.id)\
            .eq("is_favorite", True)\
            .order("created_at", desc=True)\
            .execute()
            
        recent_res = supabase.table("search_history")\
            .select("*")\
            .eq("user_id", user.id)\
            .eq("is_favorite", False)\
            .order("created_at", desc=True)\
            .limit(10)\
            .execute()
            
        return templates.TemplateResponse(
            request=request, 
            name="history.html", 
            context={
                "favorites": fav_res.data,
                "recents": recent_res.data,
                "user": user
            }
        )
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        raise HTTPException(status_code=500, detail="Error fetching history")

@router.get("/history/{search_id}", response_class=HTMLResponse)
async def view_history_detail(request: Request, search_id: str, user = Depends(require_auth)):
    """Renders a dedicated view for a past search result."""
    token = request.cookies.get("vyoris_access_token")
    supabase = get_supabase(token)
    try:
        res = supabase.table("search_history")\
            .select("*")\
            .eq("id", search_id)\
            .eq("user_id", user.id)\
            .execute()
            
        if not res.data:
            raise HTTPException(status_code=404, detail="Search not found")
            
        return templates.TemplateResponse(
            request=request, 
            name="history_detail.html", 
            context={
                "search": res.data[0],
                "user": user
            }
        )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"Error fetching history detail: {e}")
        raise HTTPException(status_code=500, detail="Error fetching history detail")

@router.post("/history/toggle-favorite/{search_id}", response_class=HTMLResponse)
async def toggle_favorite(request: Request, search_id: str, current_val: str = Form(...), user = Depends(require_auth)):
    """Toggles the favorite status of a search and returns the updated star icon HTML."""
    token = request.cookies.get("vyoris_access_token")
    supabase = get_supabase(token)
    is_favorite = current_val.lower() == 'true'
    new_val = not is_favorite
    
    try:
        supabase.table("search_history").update({"is_favorite": new_val}).eq("id", search_id).eq("user_id", user.id).execute()
        
        # Determine the color of the star icon
        color_class = "text-yellow-400 fill-current" if new_val else "text-gray-300"
        
        return HTMLResponse(content=f"""
            <button hx-post="/history/toggle-favorite/{search_id}" hx-vals='{{"current_val": "{new_val}"}}' hx-swap="outerHTML" class="focus:outline-none" title="Toggle Favorite">
                <svg class="w-6 h-6 {color_class} hover:text-yellow-500 transition-colors" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon>
                </svg>
            </button>
        """)
    except Exception as e:
        logger.error(f"Error toggling favorite: {e}")
        return HTMLResponse(content="Error", status_code=500)
