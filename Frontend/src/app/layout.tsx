// src/app/layout.tsx
import './globals.css';
import type { ReactNode } from 'react';
import AuthProvider from './AuthProvider';

export const metadata = {
  title: 'MiTiendaGPT',
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="es">
      <body className="flex h-screen bg-gray-50">
        <AuthProvider>
          <aside className="w-64 bg-neutral-900 text-white p-4 text-xl font-semibold">
            MiTiendaGPT
          </aside>
          <main className="flex-1 overflow-auto bg-gray-100 p-6">
            {children}
          </main>
        </AuthProvider>
      </body>
    </html>
  );
}
