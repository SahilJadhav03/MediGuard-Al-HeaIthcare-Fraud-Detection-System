/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                dark: {
                    bg: '#141416',
                    panel: '#1C1C21',
                    card: '#24242B',
                    border: '#33333C'
                },
                brand: {
                    primary: '#6C5CE7',
                    secondary: '#8C7EEB',
                    success: '#00B894',
                    danger: '#FF7675',
                    warning: '#FDCB6E'
                }
            },
            fontFamily: {
                sans: ['Inter', 'sans-serif'],
            }
        },
    },
    plugins: [],
}
