import axios from "axios";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export type ApiClientError = Error & {
    statusCode?: number;
    code?: string;
    isNetworkError?: boolean;
    isCanceled?: boolean;
};

export const api = axios.create({
    baseURL: API_URL,
    withCredentials: true,
    headers: {
        "Content-Type": "application/json",
    },
});

api.interceptors.response.use(
    (response) => response,
    (error) => {
        if (error.response?.status === 401) {
            // Redirect to login if 401 Unauthorized
            if (typeof window !== "undefined") {
                window.location.href = "/auth/login";
            }
        }
        const message = error.response?.data?.detail || error.message || "An error occurred";
        const enhancedError = new Error(message) as ApiClientError;
        enhancedError.statusCode = error.response?.status;
        enhancedError.code = error.code;
        enhancedError.isNetworkError = !error.response;
        enhancedError.isCanceled = axios.isCancel(error) || error.code === "ERR_CANCELED";
        return Promise.reject(enhancedError);
    }
);
