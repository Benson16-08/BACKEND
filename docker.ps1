param(
    [string]$Command = 'help'
)

function Show-Help {
    Write-Host "Usage: .\docker.ps1 [build|up|down|logs|ps|help]"
    Write-Host "  build  - Build the backend Docker image"
    Write-Host "  up     - Start services with docker compose"
    Write-Host "  down   - Stop and remove services"
    Write-Host "  logs   - Follow API logs"
    Write-Host "  ps     - List compose services"
}

switch ($Command.ToLower()) {
    'build' {
        docker build -t mediassist-backend .
        break
    }
    'up' {
        docker compose up --build
        break
    }
    'down' {
        docker compose down
        break
    }
    'logs' {
        docker compose logs -f api
        break
    }
    'ps' {
        docker compose ps
        break
    }
    default {
        Show-Help
        break
    }
}
