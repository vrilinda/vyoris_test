from fastapi import APIRouter, Request, Response, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from supabase import create_client, Client, ClientOptions
import logging
from typing import Optional

from src.config import config

logger = logging.getLogger("AuthRouter")
router = APIRouter()
templates = Jinja2Templates(directory="src/templates")

def get_supabase(token: Optional[str] = None) -> Client:
    if token:
        return create_client(
            config.supabase_url, 
            config.supabase_key, 
            options=ClientOptions(headers={"Authorization": f"Bearer {token}"})
        )
    return create_client(config.supabase_url, config.supabase_key)

async def get_current_user(request: Request):
    """Dependency to retrieve the authenticated user from the HTTP-only cookie."""
    token = request.cookies.get("vyoris_access_token")
    if not token:
        return None
    try:
        # We don't necessarily need the token in get_supabase here because get_user(token) works with anon client
        supabase = get_supabase()
        user_resp = supabase.auth.get_user(token)
        return user_resp.user
    except Exception as e:
        logger.error(f"Error validating user token: {e}")
        return None

async def require_auth(request: Request):
    """Dependency that enforces authentication."""
    user = await get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user

@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Renders the login page."""
    user = await get_current_user(request)
    if user:
        # If already logged in, redirect to home
        return HTMLResponse(content="<script>window.location.href='/';</script>")
    return templates.TemplateResponse(request=request, name="login.html")

@router.post("/auth/send-otp", response_class=HTMLResponse)
async def send_otp(request: Request, email: str = Form(...)):
    """Triggers the Supabase passwordless email OTP."""
    supabase = get_supabase()
    try:
        supabase.auth.sign_in_with_otp({"email": email})
        # Return the partial form for entering the OTP
        return templates.TemplateResponse(request=request, name="login_otp.html", context={"email": email})
    except Exception as e:
        logger.error(f"Error sending OTP: {e}")
        return HTMLResponse(content=f"<div class='p-3 bg-red-100 text-red-700 rounded-md'>Error sending OTP: {e}. Please try again.</div>", status_code=400)

@router.post("/auth/verify-otp", response_class=HTMLResponse)
async def verify_otp(request: Request, response: Response, email: str = Form(...), token: str = Form(...)):
    """Verifies the OTP and issues a secure session cookie."""
    supabase = get_supabase()
    try:
        res = supabase.auth.verify_otp({"email": email, "token": token, "type": "email"})
        if res.session:
            # Upsert the user profile in our database
            try:
                supabase.table("profiles").upsert({
                    "id": res.user.id,
                    "email": email
                }).execute()
            except Exception as profile_err:
                logger.error(f"Failed to upsert profile: {profile_err}")

            # Return an empty response but instruct HTMX to redirect to the home page
            html_res = HTMLResponse(content="Login successful. Redirecting...")
            html_res.headers["HX-Redirect"] = "/"
            
            # Set HTTP-only secure cookie
            html_res.set_cookie(
                key="vyoris_access_token",
                value=res.session.access_token,
                httponly=True,
                max_age=res.session.expires_in,
                samesite="lax",
                secure=False # Should be True in production with HTTPS
            )
            return html_res
        else:
            return HTMLResponse(content=f"<div class='p-3 bg-red-100 text-red-700 rounded-md mt-2'>Invalid OTP.</div>", status_code=400)
    except Exception as e:
        logger.error(f"Error verifying OTP: {e}")
        return HTMLResponse(content=f"<div class='p-3 bg-red-100 text-red-700 rounded-md mt-2'>Error verifying OTP: {str(e)}</div>", status_code=400)

@router.get("/auth/logout", response_class=HTMLResponse)
async def logout(response: Response):
    """Clears the session cookie and redirects home."""
    html_res = HTMLResponse(content="<script>window.location.href='/';</script>")
    html_res.delete_cookie("vyoris_access_token")
    return html_res
