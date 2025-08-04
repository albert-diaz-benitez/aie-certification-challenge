# Makefile for email crawler application

# Variables
COMPOSE_FILE=docker-compose.yml
APP_NAME=email-crawler
QDRANT_NAME=qdrant
CONTAINER_PREFIX=aie-certification-challenge

.PHONY: help build up down restart logs ps clean test

# Start all services in detached mode
up:
	@echo "Starting all services..."
	docker compose -f $(COMPOSE_FILE) up -d

# Stop and remove all containers
down:
	@echo "Stopping all services..."
	docker compose -f $(COMPOSE_FILE) down

# Show logs from all services
logs:
	docker compose -f $(COMPOSE_FILE) logs -f

# Show logs from the email crawler application
logs-app:
	docker compose -f $(COMPOSE_FILE) logs -f $(APP_NAME)

# Show logs from the Qdrant service
logs-qdrant:
	docker compose -f $(COMPOSE_FILE) logs -f $(QDRANT_NAME)

# List running containers
ps:
	docker compose -f $(COMPOSE_FILE) ps

# Clean up everything (containers, volumes, images)
clean:
	@echo "Removing all containers, volumes, and images..."
	docker compose -f $(COMPOSE_FILE) down -v --rmi all

export-dependencies:
	uv export --format requirements-txt --no-dev --output-file requirements.txt --prerelease=allow
	uv export --format requirements.txt --output-file requirements-dev.txt --prerelease=allow
