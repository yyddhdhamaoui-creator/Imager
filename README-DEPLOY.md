# Imager — website + persistent daily limit

The ZIP contains two separately deployed parts:

- `website/`: the Next.js website. Keep its existing visual design; the verdict heading remains 30% larger.
- `backend/`: FastAPI inference API with a persistent daily quota stored in PostgreSQL.

## What the limit does

- Default: 10 successful analysis attempts per visitor/IP pair per UTC day.
- The quota is enforced by the API, not just the browser.
- A random visitor ID is stored in browser localStorage so closing/reopening the page does not reset the ID.
- The database stores hashes of the visitor ID and IP, not the raw values.
- Clearing browser storage or changing browsers can create a new visitor ID; IP pairing provides an additional obstacle but is not perfect abuse prevention.

## Deploy the backend on Railway

1. Put the contents of `backend/` in the repository/folder used by the Railway API service.
2. Add a Railway PostgreSQL database to the project.
3. Ensure the backend service has `DATABASE_URL` connected to the PostgreSQL service (Railway usually offers a reference variable).
4. Add environment variables:
   - `MODEL_REPO=Imagers/Imager-1`
   - `RATE_LIMIT_SECRET=<a long random secret>`
   - `DAILY_LIMIT=10`
   - `MAX_IMAGE_MB=10`
   - `ALLOWED_ORIGINS=*` temporarily; for production replace `*` with your exact website origin.
5. Deploy and check `/health`.

Important: the code uses Transformers `AutoImageProcessor` and `AutoModelForImageClassification`. If Imager-1 is not compatible with that Transformers loading method, model startup will fail and this needs a model-specific loader. Also, the model may exceed the Railway service's memory limit.

## Deploy the website

Deploy the `website/` folder as the same Next.js app as before. Its API URL is set in `website/app/page.js` to:

`https://validio-production.up.railway.app/predict`

If your website is hosted on a different domain, set `ALLOWED_ORIGINS` to that exact origin instead of `*`.

## Endpoint

`POST /predict` accepts multipart form data:
- `file`: image file
- `visitor_id`: persistent browser visitor ID

Responses with an exhausted quota use HTTP `429`.
