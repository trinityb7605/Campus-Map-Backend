# Campus-Map-Backend

CS Project 1 - Interactive Campus Map backend.

## API

The service returns the JSON datasets from these endpoints:

- `GET /buildings`
- `GET /parking`

When deploying on Render, set the service root directory to
`backend/render_git/Campus-Map-Backend` and use the included `render.yaml`.
The frontend uses `https://mobile-courses-api.onrender.com` by default. Set
`EXPO_PUBLIC_API_URL` in the frontend environment to the service's public base
URL if the deployed Render URL differs.
