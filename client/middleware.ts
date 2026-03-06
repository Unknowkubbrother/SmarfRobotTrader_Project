import { NextResponse } from 'next/server'
import type { NextRequest } from 'next/server'

const AUTH_ROUTE_PREFIX = '/auth'

export function middleware(request: NextRequest) {
    const token = request.cookies.get('access_token')?.value
    const { pathname, search } = request.nextUrl

    const isAuthRoute =
        pathname === AUTH_ROUTE_PREFIX || pathname.startsWith(`${AUTH_ROUTE_PREFIX}/`)
    const isPublicRoute = pathname === '/' || isAuthRoute

    if (!token && !isPublicRoute) {
        const loginUrl = new URL('/auth/login', request.url)
        loginUrl.searchParams.set('next', `${pathname}${search}`)
        return NextResponse.redirect(loginUrl)
    }

    return NextResponse.next()
}

export const config = {
    matcher: [
        '/((?!api|_next/static|_next/image|favicon.ico|robots.txt|sitemap.xml|.*\\..*).*)',
    ],
}
