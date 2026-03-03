CREATE DATABASE smarfrobottrader;

USE smarfrobottrader;

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ==============================================================
-- 1. Create ENUM Types
-- ==============================================================
CREATE TYPE role_enum AS ENUM ('user', 'admin');
CREATE TYPE status_enum AS ENUM ('active', 'banned', 'pending');
CREATE TYPE ticket_status_enum AS ENUM ('open', 'in_progress', 'resolved', 'closed');
CREATE TYPE order_type_enum AS ENUM ('buy', 'sell');
CREATE TYPE risk_enum AS ENUM ('low', 'medium', 'high');
CREATE TYPE container_status_enum AS ENUM ('running', 'stopped', 'error', 'starting');
CREATE TYPE fee_type_enum AS ENUM ('percentage', 'fixed');
CREATE TYPE sub_status_enum AS ENUM ('active', 'past_due', 'canceled');
CREATE TYPE invoice_status_enum AS ENUM ('paid', 'pending', 'skipped', 'failed');

-- ==============================================================
-- 2. User Management & Auth
-- ==============================================================
CREATE TABLE "users" (
  "user_id" UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  "username" VARCHAR(50) NOT NULL,
  "email" VARCHAR(100) UNIQUE NOT NULL,
  "password" VARCHAR(255),
  "avatar_url" VARCHAR(255),
  "recovery_email" VARCHAR(100),
  "google_auth_id" VARCHAR(100),
  "stripe_customer_id" VARCHAR(100),
  "role" role_enum DEFAULT 'user',
  "status" status_enum DEFAULT 'active',
  "is_onboarding_completed" BOOLEAN DEFAULT FALSE,
  "created_at" TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
  "updated_at" TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE "user_payment_methods" (
  "method_id" UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  "user_id" UUID NOT NULL REFERENCES "users" ("user_id") ON DELETE CASCADE,
  "provider_method_id" VARCHAR(100), -- Stripe Method ID
  "type" VARCHAR(20), -- credit_card, paypal
  "card_last4" VARCHAR(4),
  "card_brand" VARCHAR(20),
  "expiry_month" INT,
  "expiry_year" INT,
  "is_active" BOOLEAN DEFAULT TRUE,
  "is_default" BOOLEAN DEFAULT FALSE
);

CREATE TABLE "activities_logs" (
  "log_id" UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  "user_id" UUID REFERENCES "users" ("user_id") ON DELETE SET NULL,
  "ip_address" VARCHAR(45),
  "device_info" VARCHAR(255),
  "topic" VARCHAR(100),
  "detail" TEXT,
  "created_at" TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- ==============================================================
-- 3. Notification & Support
-- ==============================================================
CREATE TABLE "notification_configs" (
  "user_id" UUID PRIMARY KEY REFERENCES "users" ("user_id") ON DELETE CASCADE,
  "line_notify_token" VARCHAR(255),
  "discord_webhook_url" VARCHAR(255),
  "email_notification_enable" BOOLEAN DEFAULT TRUE,
  
  -- Alert Settings (Nullable)
  "alert_margin_level_threshold" NUMERIC(10, 2),
  "alert_profit_target" NUMERIC(10, 2),
  "alert_loss_limit" NUMERIC(10, 2),
  
  "enable_weekly_summary" BOOLEAN DEFAULT TRUE,
  "enable_monthly_summary" BOOLEAN DEFAULT TRUE,
  "updated_at" TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE "notifications" (
  "notification_id" UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  "user_id" UUID NOT NULL REFERENCES "users" ("user_id") ON DELETE CASCADE,
  "title" VARCHAR(100),
  "message" TEXT,
  "related_link" VARCHAR(255),
  "is_read" BOOLEAN DEFAULT FALSE,
  "created_at" TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE "support_tickets" (
  "ticket_id" UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  "user_id" UUID NOT NULL REFERENCES "users" ("user_id") ON DELETE CASCADE,
  "subject" VARCHAR(100),
  "message" TEXT,
  "status" ticket_status_enum DEFAULT 'open',
  "created_at" TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
  "updated_at" TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- ==============================================================
-- 4. Trading Core
-- ==============================================================
CREATE TABLE "trading_accounts" (
  "account_id" UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  "user_id" UUID NOT NULL REFERENCES "users" ("user_id") ON DELETE CASCADE,
  "broker_name" VARCHAR(100),
  "server_name" VARCHAR(100),
  "mt5_login_id" VARCHAR(50),
  "mt5_password" VARCHAR(255), -- Should be encrypted
  "record_status" VARCHAR(20) NOT NULL DEFAULT 'active',
  "deleted_at" TIMESTAMPTZ,
  
  -- Real-time Status
  "balance" NUMERIC(15, 2) DEFAULT 0,
  "equity" NUMERIC(15, 2) DEFAULT 0,
  "leverage" INT,
  "margin" NUMERIC(15, 2) DEFAULT 0,
  "margin_free" NUMERIC(15, 2) DEFAULT 0,
  "margin_level" NUMERIC(10, 2) DEFAULT 0,
  
  "created_at" TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
  "updated_at" TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE "orders_history" (
  "ticket_id" BIGINT PRIMARY KEY, -- MT5 Ticket ID
  "account_id" UUID NOT NULL REFERENCES "trading_accounts" ("account_id") ON DELETE CASCADE,
  "bot_instance_id" INT, -- Links to specific bot
  
  "symbol" VARCHAR(20),
  "type" order_type_enum,
  "volume" NUMERIC(10, 2),
  "open_price" NUMERIC(10, 5),
  "close_price" NUMERIC(10, 5),
  "open_time" TIMESTAMPTZ,
  "close_time" TIMESTAMPTZ,
  
  -- Financials
  "commission" NUMERIC(10, 2) DEFAULT 0,
  "swap" NUMERIC(10, 2) DEFAULT 0,
  "profit" NUMERIC(10, 2) DEFAULT 0,
  
  "status" VARCHAR(20) -- closed, canceled
);

-- Index for querying orders by bot
CREATE INDEX "idx_orders_bot_instance" ON "orders_history" ("account_id", "bot_instance_id");

CREATE TABLE "daily_aggregates" (
  "id" UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  "account_id" UUID NOT NULL REFERENCES "trading_accounts" ("account_id") ON DELETE CASCADE,
  "date" DATE NOT NULL,
  "daily_net_profit" NUMERIC(15, 2),
  "total_trades" INT,
  
  UNIQUE ("account_id", "date")
);

CREATE TABLE "trading_journals" (
  "journal_id" UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  "ticket_id" BIGINT REFERENCES "orders_history" ("ticket_id") ON DELETE CASCADE,
  "trade_rationale" TEXT,
  "mistake_lesson" TEXT,
  "attachment_urls" JSONB, -- Postgres JSONB for better performance
  "tags" JSONB,
  "created_at" TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
  "updated_at" TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- ==============================================================
-- 5. Bot Management & Docker
-- ==============================================================
CREATE TABLE "bot_versions" (
  "model_id" SERIAL PRIMARY KEY,
  "label" VARCHAR(100),
  "docker_image_id" VARCHAR(255),
  "version_tag" VARCHAR(20),
  "symbol" VARCHAR(20),
  "timeframe" VARCHAR(10),
  "release_notes" TEXT,
  "release_date" TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE "bot_configurations" (
  "config_id" UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  "account_id" UUID NOT NULL REFERENCES "trading_accounts" ("account_id") ON DELETE CASCADE,
  "model_id" INT NOT NULL REFERENCES "bot_versions" ("model_id"),
  
  -- Multi-Bot Identifiers
  "bot_instance_id" INT NOT NULL, -- Mapped to MT5 Magic Number
  
  "risk_level" risk_enum,
  "trading_schedule" JSONB,
  "is_active" BOOLEAN DEFAULT FALSE,
  "record_status" VARCHAR(20) NOT NULL DEFAULT 'active',
  "deleted_at" TIMESTAMPTZ,
  
  -- Container Control
  "docker_container_id" VARCHAR(100),
  "installed_docker_image_id" VARCHAR(255),
  "installed_version_tag" VARCHAR(50),
  "container_status" container_status_enum,
  
  "updated_at" TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
  
  -- Prevent duplicate magic numbers in same account
  UNIQUE ("account_id", "bot_instance_id")
);

ALTER TABLE "trading_accounts"
  ADD COLUMN IF NOT EXISTS "record_status" VARCHAR(20) NOT NULL DEFAULT 'active',
  ADD COLUMN IF NOT EXISTS "deleted_at" TIMESTAMPTZ;

ALTER TABLE "bot_configurations"
  ADD COLUMN IF NOT EXISTS "record_status" VARCHAR(20) NOT NULL DEFAULT 'active',
  ADD COLUMN IF NOT EXISTS "deleted_at" TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS "idx_trading_accounts_record_status" ON "trading_accounts" ("record_status");
CREATE INDEX IF NOT EXISTS "idx_bot_configurations_record_status" ON "bot_configurations" ("record_status");

-- ==============================================================
-- 6. Billing & Subscriptions
-- ==============================================================
CREATE TABLE "system_billing_config" (
  "config_id" SERIAL PRIMARY KEY,
  "default_fee_type" fee_type_enum,
  "default_fee_value" NUMERIC(10, 2),
  "default_min_threshold" NUMERIC(10, 2),
  "default_next_billing_date" DATE,
  "updated_at" TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE "subscriptions" (
  "sub_id" UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  "user_id" UUID NOT NULL REFERENCES "users" ("user_id") ON DELETE CASCADE,
  "default_payment_method_id" UUID REFERENCES "user_payment_methods" ("method_id"),
  
  "status" sub_status_enum,
  "fee_type" fee_type_enum DEFAULT 'percentage',
  "fee_value" NUMERIC(10, 2) DEFAULT 20.00,
  "min_profit_threshold" NUMERIC(10, 2),
  "next_billing_date" DATE,
  
  "created_at" TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE "invoices" (
  "invoice_id" UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  "sub_id" UUID NOT NULL REFERENCES "subscriptions" ("sub_id") ON DELETE CASCADE,
  
  "billing_start_date" DATE,
  "billing_end_date" DATE,
  
  "total_period_profit" NUMERIC(15, 2),
  "calculated_fee" NUMERIC(10, 2),
  
  "status" invoice_status_enum,
  "payment_method_used" VARCHAR(50),
  "stripe_payment_intent_id" VARCHAR(100),
  "paid_at" TIMESTAMPTZ,
  "created_at" TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
