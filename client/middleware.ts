import { NextResponse } from 'next/server'
import type { NextRequest } from 'next/server'

// Routes that do not require authentication
const publicRoutes = [
    '/auth/login',
    '/auth/register',
    '/auth/forgot-password',
    '/auth'
]

export function middleware(request: NextRequest) {
    const token = request.cookies.get('access_token')?.value
    const { pathname } = request.nextUrl

    // Check if the path is a public route
    const isPublicRoute = publicRoutes.some(route => pathname.startsWith(route))

    // If user has token and tries to access auth pages, redirect to dashboard (/)
    if (token && isPublicRoute) {
        return NextResponse.redirect(new URL('/', request.url))
    }

    // If user does NOT have token and tries to access protected route (anything not public)
    if (!token && !isPublicRoute) {
        return NextResponse.redirect(new URL('/auth/login', request.url))
    }

    return NextResponse.next()
}

// Configure matcher to run middleware on specific paths
export const config = {
    matcher: [
        /*
         * Match all request paths except for the ones starting with:
         * - api (API routes)
         * - _next/static (static files)
         * - _next/image (image optimization files)
         * - favicon.ico (favicon file)
         * - images (public images)
         */
        '/((?!api|_next/static|_next/image|favicon.ico|images).*)',
    ],
}