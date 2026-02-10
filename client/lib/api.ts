import axios from "axios";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

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
        return Promise.reject(new Error(message));
    }
);
