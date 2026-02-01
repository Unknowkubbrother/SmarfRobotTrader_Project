-- Create enum for user roles
CREATE TYPE public.app_role AS ENUM ('admin', 'user');

-- Create enum for subscription status
CREATE TYPE public.subscription_status AS ENUM ('active', 'trial', 'suspended', 'expired');

-- Create enum for bot status
CREATE TYPE public.bot_status AS ENUM ('running', 'paused', 'stopped');

-- Create enum for order type
CREATE TYPE public.order_type AS ENUM ('buy', 'sell');

-- Create enum for order status
CREATE TYPE public.order_status AS ENUM ('open', 'closed', 'pending');

-- Create enum for ticket status
CREATE TYPE public.ticket_status AS ENUM ('open', 'in_progress', 'resolved', 'closed');

-- Create enum for invoice status
CREATE TYPE public.invoice_status AS ENUM ('pending', 'paid', 'failed', 'cancelled');

-- Create enum for fee type
CREATE TYPE public.fee_type AS ENUM ('percentage', 'fixed');

-- Create profiles table (main user data)
CREATE TABLE public.profiles (
  id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  email TEXT NOT NULL,
  recovery_email TEXT,
  google_auth_id TEXT,
  status TEXT DEFAULT 'active',
  stripe_customer_id TEXT,
  is_onboarding_completed BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create user_roles table (separate for security)
CREATE TABLE public.user_roles (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  role app_role NOT NULL DEFAULT 'user',
  UNIQUE(user_id, role)
);

-- Create trading_accounts table
CREATE TABLE public.trading_accounts (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  broker_name TEXT NOT NULL,
  server_name TEXT NOT NULL,
  mt5_login_id TEXT NOT NULL,
  mt5_password TEXT NOT NULL,
  balance DECIMAL(15,2) DEFAULT 0,
  equity DECIMAL(15,2) DEFAULT 0,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create bot_versions table
CREATE TABLE public.bot_versions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  label TEXT NOT NULL,
  docker_image_id TEXT,
  release_notes TEXT,
  release_date TIMESTAMPTZ DEFAULT NOW(),
  version_tag TEXT NOT NULL,
  symbol TEXT NOT NULL DEFAULT 'XAUUSD',
  timeframe TEXT DEFAULT 'H1',
  is_active BOOLEAN DEFAULT TRUE,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create bot_configurations table
CREATE TABLE public.bot_configurations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  account_id UUID NOT NULL REFERENCES public.trading_accounts(id) ON DELETE CASCADE,
  model_id UUID REFERENCES public.bot_versions(id),
  risk_level TEXT DEFAULT 'medium',
  trading_schedule JSONB DEFAULT '{"mon":true,"tue":true,"wed":true,"thu":true,"fri":true,"sat":false,"sun":false}',
  is_active BOOLEAN DEFAULT FALSE,
  status bot_status DEFAULT 'stopped',
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create orders_history table
CREATE TABLE public.orders_history (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  account_id UUID NOT NULL REFERENCES public.trading_accounts(id) ON DELETE CASCADE,
  ticket_id TEXT NOT NULL,
  symbol TEXT NOT NULL,
  type order_type NOT NULL,
  volume DECIMAL(10,2) NOT NULL,
  open_price DECIMAL(15,5) NOT NULL,
  close_price DECIMAL(15,5),
  open_time TIMESTAMPTZ NOT NULL,
  close_time TIMESTAMPTZ,
  commission DECIMAL(10,2) DEFAULT 0,
  swap DECIMAL(10,2) DEFAULT 0,
  profit DECIMAL(15,2),
  status order_status DEFAULT 'open'
);

-- Create daily_aggregates table
CREATE TABLE public.daily_aggregates (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  account_id UUID NOT NULL REFERENCES public.trading_accounts(id) ON DELETE CASCADE,
  date DATE NOT NULL,
  daily_net_profit DECIMAL(15,2) DEFAULT 0,
  total_trades INTEGER DEFAULT 0,
  UNIQUE(account_id, date)
);

-- Create trading_journals table
CREATE TABLE public.trading_journals (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  ticket_id UUID REFERENCES public.orders_history(id) ON DELETE CASCADE,
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  trade_rationale TEXT,
  mistake_lesson TEXT,
  attachment_urls TEXT[],
  tags TEXT[],
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create notifications table
CREATE TABLE public.notifications (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  title TEXT NOT NULL,
  message TEXT NOT NULL,
  related_link TEXT,
  is_read BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create notification_configs table
CREATE TABLE public.notification_configs (
  user_id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  line_notify_token TEXT,
  discord_webhook_url TEXT,
  email_notification_enable BOOLEAN DEFAULT TRUE,
  alert_margin_level_threshold DECIMAL(5,2) DEFAULT 100,
  alert_daily_profit_target DECIMAL(15,2),
  alert_daily_loss_limit DECIMAL(15,2),
  enable_weekly_summary BOOLEAN DEFAULT TRUE,
  enable_monthly_summary BOOLEAN DEFAULT TRUE,
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create activity_logs table
CREATE TABLE public.activity_logs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  ip_address TEXT,
  device_info TEXT,
  topic TEXT NOT NULL,
  detail TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create support_tickets table
CREATE TABLE public.support_tickets (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  subject TEXT NOT NULL,
  message TEXT NOT NULL,
  status ticket_status DEFAULT 'open',
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create user_payment_methods table
CREATE TABLE public.user_payment_methods (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  provider_method_id TEXT,
  type TEXT NOT NULL,
  card_last4 TEXT,
  card_brand TEXT,
  expiry_month INTEGER,
  expiry_year INTEGER,
  is_active BOOLEAN DEFAULT TRUE,
  is_default BOOLEAN DEFAULT FALSE
);

-- Create subscriptions table
CREATE TABLE public.subscriptions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  default_payment_method_id UUID REFERENCES public.user_payment_methods(id),
  status subscription_status DEFAULT 'trial',
  fee_type fee_type DEFAULT 'percentage',
  fee_value DECIMAL(5,2) DEFAULT 20,
  min_profit_threshold DECIMAL(15,2) DEFAULT 0,
  next_billing_date DATE,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create invoices table
CREATE TABLE public.invoices (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  sub_id UUID NOT NULL REFERENCES public.subscriptions(id) ON DELETE CASCADE,
  billing_start_date DATE NOT NULL,
  billing_end_date DATE NOT NULL,
  total_period_profit DECIMAL(15,2) DEFAULT 0,
  calculated_fee DECIMAL(15,2) DEFAULT 0,
  status invoice_status DEFAULT 'pending',
  payment_method_used UUID REFERENCES public.user_payment_methods(id),
  stripe_payment_intent_id TEXT,
  paid_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create system_billing_config table (admin only)
CREATE TABLE public.system_billing_config (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  default_fee_type fee_type DEFAULT 'percentage',
  default_fee_value DECIMAL(5,2) DEFAULT 20,
  default_min_threshold DECIMAL(15,2) DEFAULT 0,
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable RLS on all tables
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.user_roles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.trading_accounts ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.bot_versions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.bot_configurations ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.orders_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.daily_aggregates ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.trading_journals ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.notifications ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.notification_configs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.activity_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.support_tickets ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.user_payment_methods ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.subscriptions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.invoices ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.system_billing_config ENABLE ROW LEVEL SECURITY;

-- Create security definer function for role checking
CREATE OR REPLACE FUNCTION public.has_role(_user_id UUID, _role app_role)
RETURNS BOOLEAN
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
  SELECT EXISTS (
    SELECT 1
    FROM public.user_roles
    WHERE user_id = _user_id
      AND role = _role
  )
$$;

-- RLS Policies for profiles
CREATE POLICY "Users can view own profile" ON public.profiles FOR SELECT USING (auth.uid() = id);
CREATE POLICY "Users can update own profile" ON public.profiles FOR UPDATE USING (auth.uid() = id);
CREATE POLICY "Users can insert own profile" ON public.profiles FOR INSERT WITH CHECK (auth.uid() = id);
CREATE POLICY "Admins can view all profiles" ON public.profiles FOR SELECT USING (public.has_role(auth.uid(), 'admin'));

-- RLS Policies for user_roles
CREATE POLICY "Users can view own roles" ON public.user_roles FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Admins can manage roles" ON public.user_roles FOR ALL USING (public.has_role(auth.uid(), 'admin'));

-- RLS Policies for trading_accounts
CREATE POLICY "Users can view own accounts" ON public.trading_accounts FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Users can insert own accounts" ON public.trading_accounts FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Users can update own accounts" ON public.trading_accounts FOR UPDATE USING (auth.uid() = user_id);
CREATE POLICY "Users can delete own accounts" ON public.trading_accounts FOR DELETE USING (auth.uid() = user_id);
CREATE POLICY "Admins can view all accounts" ON public.trading_accounts FOR SELECT USING (public.has_role(auth.uid(), 'admin'));

-- RLS Policies for bot_versions (public read, admin write)
CREATE POLICY "Anyone can view bot versions" ON public.bot_versions FOR SELECT USING (true);
CREATE POLICY "Admins can manage bot versions" ON public.bot_versions FOR ALL USING (public.has_role(auth.uid(), 'admin'));

-- RLS Policies for bot_configurations
CREATE POLICY "Users can view own configs" ON public.bot_configurations FOR SELECT 
USING (EXISTS (SELECT 1 FROM public.trading_accounts WHERE id = account_id AND user_id = auth.uid()));
CREATE POLICY "Users can insert own configs" ON public.bot_configurations FOR INSERT 
WITH CHECK (EXISTS (SELECT 1 FROM public.trading_accounts WHERE id = account_id AND user_id = auth.uid()));
CREATE POLICY "Users can update own configs" ON public.bot_configurations FOR UPDATE 
USING (EXISTS (SELECT 1 FROM public.trading_accounts WHERE id = account_id AND user_id = auth.uid()));
CREATE POLICY "Admins can view all configs" ON public.bot_configurations FOR SELECT USING (public.has_role(auth.uid(), 'admin'));

-- RLS Policies for orders_history
CREATE POLICY "Users can view own orders" ON public.orders_history FOR SELECT 
USING (EXISTS (SELECT 1 FROM public.trading_accounts WHERE id = account_id AND user_id = auth.uid()));
CREATE POLICY "Admins can view all orders" ON public.orders_history FOR SELECT USING (public.has_role(auth.uid(), 'admin'));

-- RLS Policies for daily_aggregates
CREATE POLICY "Users can view own aggregates" ON public.daily_aggregates FOR SELECT 
USING (EXISTS (SELECT 1 FROM public.trading_accounts WHERE id = account_id AND user_id = auth.uid()));
CREATE POLICY "Admins can view all aggregates" ON public.daily_aggregates FOR SELECT USING (public.has_role(auth.uid(), 'admin'));

-- RLS Policies for trading_journals
CREATE POLICY "Users can manage own journals" ON public.trading_journals FOR ALL USING (auth.uid() = user_id);

-- RLS Policies for notifications
CREATE POLICY "Users can view own notifications" ON public.notifications FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Users can update own notifications" ON public.notifications FOR UPDATE USING (auth.uid() = user_id);

-- RLS Policies for notification_configs
CREATE POLICY "Users can manage own notification config" ON public.notification_configs FOR ALL USING (auth.uid() = user_id);

-- RLS Policies for activity_logs
CREATE POLICY "Users can view own activity logs" ON public.activity_logs FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Admins can view all activity logs" ON public.activity_logs FOR SELECT USING (public.has_role(auth.uid(), 'admin'));

-- RLS Policies for support_tickets
CREATE POLICY "Users can manage own tickets" ON public.support_tickets FOR ALL USING (auth.uid() = user_id);
CREATE POLICY "Admins can view all tickets" ON public.support_tickets FOR SELECT USING (public.has_role(auth.uid(), 'admin'));
CREATE POLICY "Admins can update tickets" ON public.support_tickets FOR UPDATE USING (public.has_role(auth.uid(), 'admin'));

-- RLS Policies for user_payment_methods
CREATE POLICY "Users can manage own payment methods" ON public.user_payment_methods FOR ALL USING (auth.uid() = user_id);

-- RLS Policies for subscriptions
CREATE POLICY "Users can view own subscription" ON public.subscriptions FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Admins can manage subscriptions" ON public.subscriptions FOR ALL USING (public.has_role(auth.uid(), 'admin'));

-- RLS Policies for invoices
CREATE POLICY "Users can view own invoices" ON public.invoices FOR SELECT 
USING (EXISTS (SELECT 1 FROM public.subscriptions WHERE id = sub_id AND user_id = auth.uid()));
CREATE POLICY "Admins can manage invoices" ON public.invoices FOR ALL USING (public.has_role(auth.uid(), 'admin'));

-- RLS Policies for system_billing_config (admin only)
CREATE POLICY "Admins can manage billing config" ON public.system_billing_config FOR ALL USING (public.has_role(auth.uid(), 'admin'));
CREATE POLICY "Anyone can view billing config" ON public.system_billing_config FOR SELECT USING (true);

-- Create function to handle new user signup
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER SET search_path = public
AS $$
BEGIN
  -- Create profile
  INSERT INTO public.profiles (id, email)
  VALUES (NEW.id, NEW.email);
  
  -- Assign default user role
  INSERT INTO public.user_roles (user_id, role)
  VALUES (NEW.id, 'user');
  
  -- Create default subscription (trial)
  INSERT INTO public.subscriptions (user_id, status, next_billing_date)
  VALUES (NEW.id, 'trial', CURRENT_DATE + INTERVAL '7 days');
  
  -- Create default notification config
  INSERT INTO public.notification_configs (user_id)
  VALUES (NEW.id);
  
  RETURN NEW;
END;
$$;

-- Create trigger for new user signup
CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION public.update_updated_at_column()
RETURNS TRIGGER
LANGUAGE plpgsql
SET search_path = public
AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$;

-- Create triggers for updated_at
CREATE TRIGGER update_profiles_updated_at BEFORE UPDATE ON public.profiles FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();
CREATE TRIGGER update_trading_accounts_updated_at BEFORE UPDATE ON public.trading_accounts FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();
CREATE TRIGGER update_bot_configurations_updated_at BEFORE UPDATE ON public.bot_configurations FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();
CREATE TRIGGER update_trading_journals_updated_at BEFORE UPDATE ON public.trading_journals FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();
CREATE TRIGGER update_notification_configs_updated_at BEFORE UPDATE ON public.notification_configs FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();
CREATE TRIGGER update_support_tickets_updated_at BEFORE UPDATE ON public.support_tickets FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();
CREATE TRIGGER update_subscriptions_updated_at BEFORE UPDATE ON public.subscriptions FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();
CREATE TRIGGER update_system_billing_config_updated_at BEFORE UPDATE ON public.system_billing_config FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();