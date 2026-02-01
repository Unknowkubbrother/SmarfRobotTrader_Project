export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[]

export type Database = {
  // Allows to automatically instantiate createClient with right options
  // instead of createClient<Database, { PostgrestVersion: 'XX' }>(URL, KEY)
  __InternalSupabase: {
    PostgrestVersion: "14.1"
  }
  public: {
    Tables: {
      activity_logs: {
        Row: {
          created_at: string | null
          detail: string | null
          device_info: string | null
          id: string
          ip_address: string | null
          topic: string
          user_id: string
        }
        Insert: {
          created_at?: string | null
          detail?: string | null
          device_info?: string | null
          id?: string
          ip_address?: string | null
          topic: string
          user_id: string
        }
        Update: {
          created_at?: string | null
          detail?: string | null
          device_info?: string | null
          id?: string
          ip_address?: string | null
          topic?: string
          user_id?: string
        }
        Relationships: []
      }
      bot_configurations: {
        Row: {
          account_id: string
          bot_instance_id: string | null
          container_status: string | null
          created_at: string | null
          docker_container_id: string | null
          id: string
          is_active: boolean | null
          model_id: string | null
          risk_level: string | null
          status: Database["public"]["Enums"]["bot_status"] | null
          trading_schedule: Json | null
          updated_at: string | null
        }
        Insert: {
          account_id: string
          bot_instance_id?: string | null
          container_status?: string | null
          created_at?: string | null
          docker_container_id?: string | null
          id?: string
          is_active?: boolean | null
          model_id?: string | null
          risk_level?: string | null
          status?: Database["public"]["Enums"]["bot_status"] | null
          trading_schedule?: Json | null
          updated_at?: string | null
        }
        Update: {
          account_id?: string
          bot_instance_id?: string | null
          container_status?: string | null
          created_at?: string | null
          docker_container_id?: string | null
          id?: string
          is_active?: boolean | null
          model_id?: string | null
          risk_level?: string | null
          status?: Database["public"]["Enums"]["bot_status"] | null
          trading_schedule?: Json | null
          updated_at?: string | null
        }
        Relationships: [
          {
            foreignKeyName: "bot_configurations_account_id_fkey"
            columns: ["account_id"]
            isOneToOne: false
            referencedRelation: "trading_accounts"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "bot_configurations_model_id_fkey"
            columns: ["model_id"]
            isOneToOne: false
            referencedRelation: "bot_versions"
            referencedColumns: ["id"]
          },
        ]
      }
      bot_versions: {
        Row: {
          created_at: string | null
          docker_image_id: string | null
          id: string
          is_active: boolean | null
          label: string
          release_date: string | null
          release_notes: string | null
          symbol: string
          timeframe: string | null
          version_tag: string
        }
        Insert: {
          created_at?: string | null
          docker_image_id?: string | null
          id?: string
          is_active?: boolean | null
          label: string
          release_date?: string | null
          release_notes?: string | null
          symbol?: string
          timeframe?: string | null
          version_tag: string
        }
        Update: {
          created_at?: string | null
          docker_image_id?: string | null
          id?: string
          is_active?: boolean | null
          label?: string
          release_date?: string | null
          release_notes?: string | null
          symbol?: string
          timeframe?: string | null
          version_tag?: string
        }
        Relationships: []
      }
      daily_aggregates: {
        Row: {
          account_id: string
          daily_net_profit: number | null
          date: string
          id: string
          total_trades: number | null
        }
        Insert: {
          account_id: string
          daily_net_profit?: number | null
          date: string
          id?: string
          total_trades?: number | null
        }
        Update: {
          account_id?: string
          daily_net_profit?: number | null
          date?: string
          id?: string
          total_trades?: number | null
        }
        Relationships: [
          {
            foreignKeyName: "daily_aggregates_account_id_fkey"
            columns: ["account_id"]
            isOneToOne: false
            referencedRelation: "trading_accounts"
            referencedColumns: ["id"]
          },
        ]
      }
      invoices: {
        Row: {
          billing_end_date: string
          billing_start_date: string
          calculated_fee: number | null
          created_at: string | null
          id: string
          paid_at: string | null
          payment_method_used: string | null
          status: Database["public"]["Enums"]["invoice_status"] | null
          stripe_payment_intent_id: string | null
          sub_id: string
          total_period_profit: number | null
        }
        Insert: {
          billing_end_date: string
          billing_start_date: string
          calculated_fee?: number | null
          created_at?: string | null
          id?: string
          paid_at?: string | null
          payment_method_used?: string | null
          status?: Database["public"]["Enums"]["invoice_status"] | null
          stripe_payment_intent_id?: string | null
          sub_id: string
          total_period_profit?: number | null
        }
        Update: {
          billing_end_date?: string
          billing_start_date?: string
          calculated_fee?: number | null
          created_at?: string | null
          id?: string
          paid_at?: string | null
          payment_method_used?: string | null
          status?: Database["public"]["Enums"]["invoice_status"] | null
          stripe_payment_intent_id?: string | null
          sub_id?: string
          total_period_profit?: number | null
        }
        Relationships: [
          {
            foreignKeyName: "invoices_payment_method_used_fkey"
            columns: ["payment_method_used"]
            isOneToOne: false
            referencedRelation: "user_payment_methods"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "invoices_sub_id_fkey"
            columns: ["sub_id"]
            isOneToOne: false
            referencedRelation: "subscriptions"
            referencedColumns: ["id"]
          },
        ]
      }
      notification_configs: {
        Row: {
          alert_daily_loss_limit: number | null
          alert_daily_profit_target: number | null
          alert_margin_level_threshold: number | null
          discord_webhook_url: string | null
          email_notification_enable: boolean | null
          enable_monthly_summary: boolean | null
          enable_weekly_summary: boolean | null
          line_notify_token: string | null
          updated_at: string | null
          user_id: string
        }
        Insert: {
          alert_daily_loss_limit?: number | null
          alert_daily_profit_target?: number | null
          alert_margin_level_threshold?: number | null
          discord_webhook_url?: string | null
          email_notification_enable?: boolean | null
          enable_monthly_summary?: boolean | null
          enable_weekly_summary?: boolean | null
          line_notify_token?: string | null
          updated_at?: string | null
          user_id: string
        }
        Update: {
          alert_daily_loss_limit?: number | null
          alert_daily_profit_target?: number | null
          alert_margin_level_threshold?: number | null
          discord_webhook_url?: string | null
          email_notification_enable?: boolean | null
          enable_monthly_summary?: boolean | null
          enable_weekly_summary?: boolean | null
          line_notify_token?: string | null
          updated_at?: string | null
          user_id?: string
        }
        Relationships: []
      }
      notifications: {
        Row: {
          created_at: string | null
          id: string
          is_read: boolean | null
          message: string
          related_link: string | null
          title: string
          user_id: string
        }
        Insert: {
          created_at?: string | null
          id?: string
          is_read?: boolean | null
          message: string
          related_link?: string | null
          title: string
          user_id: string
        }
        Update: {
          created_at?: string | null
          id?: string
          is_read?: boolean | null
          message?: string
          related_link?: string | null
          title?: string
          user_id?: string
        }
        Relationships: []
      }
      orders_history: {
        Row: {
          account_id: string
          bot_instance_id: string | null
          close_price: number | null
          close_time: string | null
          commission: number | null
          id: string
          open_price: number
          open_time: string
          profit: number | null
          status: Database["public"]["Enums"]["order_status"] | null
          swap: number | null
          symbol: string
          ticket_id: string
          type: Database["public"]["Enums"]["order_type"]
          volume: number
        }
        Insert: {
          account_id: string
          bot_instance_id?: string | null
          close_price?: number | null
          close_time?: string | null
          commission?: number | null
          id?: string
          open_price: number
          open_time: string
          profit?: number | null
          status?: Database["public"]["Enums"]["order_status"] | null
          swap?: number | null
          symbol: string
          ticket_id: string
          type: Database["public"]["Enums"]["order_type"]
          volume: number
        }
        Update: {
          account_id?: string
          bot_instance_id?: string | null
          close_price?: number | null
          close_time?: string | null
          commission?: number | null
          id?: string
          open_price?: number
          open_time?: string
          profit?: number | null
          status?: Database["public"]["Enums"]["order_status"] | null
          swap?: number | null
          symbol?: string
          ticket_id?: string
          type?: Database["public"]["Enums"]["order_type"]
          volume?: number
        }
        Relationships: [
          {
            foreignKeyName: "orders_history_account_id_fkey"
            columns: ["account_id"]
            isOneToOne: false
            referencedRelation: "trading_accounts"
            referencedColumns: ["id"]
          },
        ]
      }
      profiles: {
        Row: {
          created_at: string | null
          email: string
          google_auth_id: string | null
          id: string
          is_onboarding_completed: boolean | null
          recovery_email: string | null
          status: string | null
          stripe_customer_id: string | null
          updated_at: string | null
        }
        Insert: {
          created_at?: string | null
          email: string
          google_auth_id?: string | null
          id: string
          is_onboarding_completed?: boolean | null
          recovery_email?: string | null
          status?: string | null
          stripe_customer_id?: string | null
          updated_at?: string | null
        }
        Update: {
          created_at?: string | null
          email?: string
          google_auth_id?: string | null
          id?: string
          is_onboarding_completed?: boolean | null
          recovery_email?: string | null
          status?: string | null
          stripe_customer_id?: string | null
          updated_at?: string | null
        }
        Relationships: []
      }
      subscriptions: {
        Row: {
          created_at: string | null
          default_payment_method_id: string | null
          fee_type: Database["public"]["Enums"]["fee_type"] | null
          fee_value: number | null
          id: string
          min_profit_threshold: number | null
          next_billing_date: string | null
          status: Database["public"]["Enums"]["subscription_status"] | null
          updated_at: string | null
          user_id: string
        }
        Insert: {
          created_at?: string | null
          default_payment_method_id?: string | null
          fee_type?: Database["public"]["Enums"]["fee_type"] | null
          fee_value?: number | null
          id?: string
          min_profit_threshold?: number | null
          next_billing_date?: string | null
          status?: Database["public"]["Enums"]["subscription_status"] | null
          updated_at?: string | null
          user_id: string
        }
        Update: {
          created_at?: string | null
          default_payment_method_id?: string | null
          fee_type?: Database["public"]["Enums"]["fee_type"] | null
          fee_value?: number | null
          id?: string
          min_profit_threshold?: number | null
          next_billing_date?: string | null
          status?: Database["public"]["Enums"]["subscription_status"] | null
          updated_at?: string | null
          user_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "subscriptions_default_payment_method_id_fkey"
            columns: ["default_payment_method_id"]
            isOneToOne: false
            referencedRelation: "user_payment_methods"
            referencedColumns: ["id"]
          },
        ]
      }
      support_tickets: {
        Row: {
          created_at: string | null
          id: string
          message: string
          status: Database["public"]["Enums"]["ticket_status"] | null
          subject: string
          updated_at: string | null
          user_id: string
        }
        Insert: {
          created_at?: string | null
          id?: string
          message: string
          status?: Database["public"]["Enums"]["ticket_status"] | null
          subject: string
          updated_at?: string | null
          user_id: string
        }
        Update: {
          created_at?: string | null
          id?: string
          message?: string
          status?: Database["public"]["Enums"]["ticket_status"] | null
          subject?: string
          updated_at?: string | null
          user_id?: string
        }
        Relationships: []
      }
      system_billing_config: {
        Row: {
          default_fee_type: Database["public"]["Enums"]["fee_type"] | null
          default_fee_value: number | null
          default_min_threshold: number | null
          id: string
          updated_at: string | null
        }
        Insert: {
          default_fee_type?: Database["public"]["Enums"]["fee_type"] | null
          default_fee_value?: number | null
          default_min_threshold?: number | null
          id?: string
          updated_at?: string | null
        }
        Update: {
          default_fee_type?: Database["public"]["Enums"]["fee_type"] | null
          default_fee_value?: number | null
          default_min_threshold?: number | null
          id?: string
          updated_at?: string | null
        }
        Relationships: []
      }
      trading_accounts: {
        Row: {
          balance: number | null
          broker_name: string
          created_at: string | null
          equity: number | null
          id: string
          mt5_login_id: string
          mt5_password: string
          server_name: string
          updated_at: string | null
          user_id: string
        }
        Insert: {
          balance?: number | null
          broker_name: string
          created_at?: string | null
          equity?: number | null
          id?: string
          mt5_login_id: string
          mt5_password: string
          server_name: string
          updated_at?: string | null
          user_id: string
        }
        Update: {
          balance?: number | null
          broker_name?: string
          created_at?: string | null
          equity?: number | null
          id?: string
          mt5_login_id?: string
          mt5_password?: string
          server_name?: string
          updated_at?: string | null
          user_id?: string
        }
        Relationships: []
      }
      trading_journals: {
        Row: {
          attachment_urls: string[] | null
          created_at: string | null
          id: string
          mistake_lesson: string | null
          tags: string[] | null
          ticket_id: string | null
          trade_rationale: string | null
          updated_at: string | null
          user_id: string
        }
        Insert: {
          attachment_urls?: string[] | null
          created_at?: string | null
          id?: string
          mistake_lesson?: string | null
          tags?: string[] | null
          ticket_id?: string | null
          trade_rationale?: string | null
          updated_at?: string | null
          user_id: string
        }
        Update: {
          attachment_urls?: string[] | null
          created_at?: string | null
          id?: string
          mistake_lesson?: string | null
          tags?: string[] | null
          ticket_id?: string | null
          trade_rationale?: string | null
          updated_at?: string | null
          user_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "trading_journals_ticket_id_fkey"
            columns: ["ticket_id"]
            isOneToOne: false
            referencedRelation: "orders_history"
            referencedColumns: ["id"]
          },
        ]
      }
      user_payment_methods: {
        Row: {
          card_brand: string | null
          card_last4: string | null
          expiry_month: number | null
          expiry_year: number | null
          id: string
          is_active: boolean | null
          is_default: boolean | null
          provider_method_id: string | null
          type: string
          user_id: string
        }
        Insert: {
          card_brand?: string | null
          card_last4?: string | null
          expiry_month?: number | null
          expiry_year?: number | null
          id?: string
          is_active?: boolean | null
          is_default?: boolean | null
          provider_method_id?: string | null
          type: string
          user_id: string
        }
        Update: {
          card_brand?: string | null
          card_last4?: string | null
          expiry_month?: number | null
          expiry_year?: number | null
          id?: string
          is_active?: boolean | null
          is_default?: boolean | null
          provider_method_id?: string | null
          type?: string
          user_id?: string
        }
        Relationships: []
      }
      user_roles: {
        Row: {
          id: string
          role: Database["public"]["Enums"]["app_role"]
          user_id: string
        }
        Insert: {
          id?: string
          role?: Database["public"]["Enums"]["app_role"]
          user_id: string
        }
        Update: {
          id?: string
          role?: Database["public"]["Enums"]["app_role"]
          user_id?: string
        }
        Relationships: []
      }
    }
    Views: {
      [_ in never]: never
    }
    Functions: {
      has_role: {
        Args: {
          _role: Database["public"]["Enums"]["app_role"]
          _user_id: string
        }
        Returns: boolean
      }
    }
    Enums: {
      app_role: "admin" | "user"
      bot_status: "running" | "paused" | "stopped"
      fee_type: "percentage" | "fixed"
      invoice_status: "pending" | "paid" | "failed" | "cancelled"
      order_status: "open" | "closed" | "pending"
      order_type: "buy" | "sell"
      subscription_status: "active" | "trial" | "suspended" | "expired"
      ticket_status: "open" | "in_progress" | "resolved" | "closed"
    }
    CompositeTypes: {
      [_ in never]: never
    }
  }
}

type DatabaseWithoutInternals = Omit<Database, "__InternalSupabase">

type DefaultSchema = DatabaseWithoutInternals[Extract<keyof Database, "public">]

export type Tables<
  DefaultSchemaTableNameOrOptions extends
    | keyof (DefaultSchema["Tables"] & DefaultSchema["Views"])
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof (DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
        DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Views"])
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? (DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
      DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Views"])[TableName] extends {
      Row: infer R
    }
    ? R
    : never
  : DefaultSchemaTableNameOrOptions extends keyof (DefaultSchema["Tables"] &
        DefaultSchema["Views"])
    ? (DefaultSchema["Tables"] &
        DefaultSchema["Views"])[DefaultSchemaTableNameOrOptions] extends {
        Row: infer R
      }
      ? R
      : never
    : never

export type TablesInsert<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Insert: infer I
    }
    ? I
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Insert: infer I
      }
      ? I
      : never
    : never

export type TablesUpdate<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Update: infer U
    }
    ? U
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Update: infer U
      }
      ? U
      : never
    : never

export type Enums<
  DefaultSchemaEnumNameOrOptions extends
    | keyof DefaultSchema["Enums"]
    | { schema: keyof DatabaseWithoutInternals },
  EnumName extends DefaultSchemaEnumNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"]
    : never = never,
> = DefaultSchemaEnumNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"][EnumName]
  : DefaultSchemaEnumNameOrOptions extends keyof DefaultSchema["Enums"]
    ? DefaultSchema["Enums"][DefaultSchemaEnumNameOrOptions]
    : never

export type CompositeTypes<
  PublicCompositeTypeNameOrOptions extends
    | keyof DefaultSchema["CompositeTypes"]
    | { schema: keyof DatabaseWithoutInternals },
  CompositeTypeName extends PublicCompositeTypeNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"]
    : never = never,
> = PublicCompositeTypeNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"][CompositeTypeName]
  : PublicCompositeTypeNameOrOptions extends keyof DefaultSchema["CompositeTypes"]
    ? DefaultSchema["CompositeTypes"][PublicCompositeTypeNameOrOptions]
    : never

export const Constants = {
  public: {
    Enums: {
      app_role: ["admin", "user"],
      bot_status: ["running", "paused", "stopped"],
      fee_type: ["percentage", "fixed"],
      invoice_status: ["pending", "paid", "failed", "cancelled"],
      order_status: ["open", "closed", "pending"],
      order_type: ["buy", "sell"],
      subscription_status: ["active", "trial", "suspended", "expired"],
      ticket_status: ["open", "in_progress", "resolved", "closed"],
    },
  },
} as const
