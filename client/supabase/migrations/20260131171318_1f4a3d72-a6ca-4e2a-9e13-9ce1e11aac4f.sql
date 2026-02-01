-- Add new columns to bot_configurations
ALTER TABLE bot_configurations
ADD COLUMN bot_instance_id text UNIQUE,
ADD COLUMN docker_container_id text,
ADD COLUMN container_status text DEFAULT 'stopped';

-- Add bot_instance_id to orders_history
ALTER TABLE orders_history
ADD COLUMN bot_instance_id text;

-- Add RLS policy for deleting bot configurations
CREATE POLICY "Users can delete own configs"
ON bot_configurations FOR DELETE
USING (
  EXISTS (
    SELECT 1 FROM trading_accounts
    WHERE trading_accounts.id = bot_configurations.account_id
    AND trading_accounts.user_id = auth.uid()
  )
);