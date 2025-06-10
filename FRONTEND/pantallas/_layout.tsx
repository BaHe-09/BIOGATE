import { Slot, useRouter, useSegments } from 'expo-router';
import { useEffect, useState } from 'react';
import { ThemeProvider } from '../context/ThemeContext';

export default function RootLayout() {
  const router = useRouter();
  const segments = useSegments();
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    if (!Array.isArray(segments[0]) || segments[0].length === 0) return;
    setIsReady(true);
  }, [segments]);

  useEffect(() => {
    if (!isReady) return;
    const currentRoute = segments[0]?.[0] || '';
    if (currentRoute !== 'login') {
      router.replace('/login');
    }
  }, [isReady, segments]);

  return (
    <ThemeProvider>
      <Slot />
    </ThemeProvider>
  );
}
