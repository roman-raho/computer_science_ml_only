import os
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel
from fastapi import FastAPI, Depends, HTTPException, status
from supabase import create_client, Client

url = "https://xotakuqckqcikadxnpsl.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InhvdGFrdXFja3FjaWthZHhucHNsIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NzIzNjk0OSwiZXhwIjoyMDcyODEyOTQ5fQ.V6NsstER0GnVU7jUEhJ5QBBk8RaNrZM8580ygRoRW5o"
supabase: Client = create_client(url, key)

class ValidationOut(BaseModel):
  artwork_id: str
  ensemble_value: float
  lower: float
  upper: float
  confidence: float
  fetched_at: Optional[datetime]

# database access

class ValuationDatabase:
  def __init__(self, supab_client: Client):
    self.supab = supab_client

  # get by id
  def get_by_id(self, artwork_id: str) -> Optional[dict]:
    db = self.supab.table("validation_current").select("*").eq("artwork_id", artwork_id).maybe_single().execute()

    # did some rearch and supabase returns sometimes .data so i will handle both
    data = None

    if db is None:
      return None
        
    data = getattr(db, "data", None)
    if data is None and isinstance(db, dict):
      data = db.get("data")
    
    if not data:
      return None
    
    if isinstance(data, list):
      return data[0]
    return data
  
  # batch list artworks
  def list(self, limit: int = 100):
    db = self.supab.table("validation_current").select("*").limit(limit)
    if isinstance(db, dict):
      return db.get("data", [])
    return getattr(db, "data", [])
  

class ValuationFetch:
  def __init__(self, db: ValuationDatabase):
    self.db = db

  def fetch(self, artwork_id: str) -> ValidationOut:
    row = self.db.get_by_id(artwork_id)
    if not row:
      raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="artwork_id invalid")
    
    return ValidationOut(
      artwork_id=str(row["artwork_id"]),
      ensemble_value=float(row["ensemble_value"]),
      lower=float(row["lower"]),
      upper=float(row["upper"]),
      confidence=float(row.get("confidence", 0.0)),
      fetched_at=row.get("fetched_at")
    )
  
  def list(self, limit: int = 100) -> List[ValidationOut]:
    rows = self.db.list(limit=limit)
    return [self.fetch(r["artwork_id"]) for r in rows]
  

def get_db() -> ValuationDatabase:
  return ValuationDatabase(supabase)

def get_service(db: ValuationDatabase = Depends(get_db)):
  return ValuationFetch(db)

app = FastAPI(title="VALUATION API")

@app.get("/valuation/{artwork_id}", response_model=ValidationOut)
def get_valuation(artwork_id: str, svc: ValuationFetch = Depends(get_service)):
  return svc.fetch(artwork_id)

@app.get("valuations", response_model=List[ValidationOut])
def list_valuations(limit: int = 100, svc: ValuationFetch = Depends(get_service)):
  return svc.list(limit=limit)
