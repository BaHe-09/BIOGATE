import { Ionicons } from '@expo/vector-icons';
import axios from 'axios';
import { useRouter } from 'expo-router';
import * as WebBrowser from 'expo-web-browser';
import { useEffect, useState } from 'react';
import {
  ActivityIndicator,
  Image,
  RefreshControl,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View
} from 'react-native';
import { useTheme } from '../context/ThemeContext';

// Definir interfaces para los tipos
interface Reporte {
  id: number;
  titulo: string;
  descripcion: string;
  tipo: string;
  severidad: string;
  estado: string;
  fecha: string;
  hora: string;
  nombre: string;
  ubicacion: string;
  evidencias: string[];
}

export default function RegistroRepScreen() {
  const { temaOscuro } = useTheme();
  const router = useRouter();
  const dynamicStyles = temaOscuro ? darkStyles : lightStyles;
  const [reportes, setReportes] = useState<Reporte[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState<boolean>(false);

  const fetchReportes = async () => {
    try {
      // Reemplaza con tu URL de API
      const response = await axios.get('https://render-biogate.onrender.com/reportes/');
      const transformedData: Reporte[] = response.data.map((item: any) => ({
        id: item.id_reporte,
        titulo: item.titulo,
        descripcion: item.descripcion,
        tipo: item.tipo_reporte,
        severidad: item.severidad || 'No especificada',
        estado: item.estado,
        fecha: item.fecha,
        hora: item.hora,
        nombre: item.nombre,
        ubicacion: item.ubicacion,
        evidencias: item.evidencias || []
      }));
      setReportes(transformedData);
      setError(null);
    } catch (err) {
      setError('Error al cargar los reportes. Intenta nuevamente.');
      console.error('Error fetching reports:', err);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const onRefresh = () => {
    setRefreshing(true);
    fetchReportes();
  };

  useEffect(() => {
    fetchReportes();
  }, []);

  const handleOpenEvidence = async (url: string) => {
    try {
      await WebBrowser.openBrowserAsync(url);
    } catch (error) {
      console.error('Error opening evidence:', error);
    }
  };

  if (loading && !refreshing) {
    return (
      <SafeAreaView style={[{ flex: 1 }, dynamicStyles.container]}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#4a90e2" />
          <Text style={[dynamicStyles.text, { marginTop: 10 }]}>Cargando reportes...</Text>
        </View>
      </SafeAreaView>
    );
  }

  if (error && !refreshing) {
    return (
      <SafeAreaView style={[{ flex: 1 }, dynamicStyles.container]}>
        <TouchableOpacity
          onPress={() => router.replace('/elegir')}
          style={[styles.backButton, temaOscuro ? styles.backDark : styles.backLight]}
        >
          <Ionicons name="arrow-back" size={24} color={temaOscuro ? '#fff' : '#000'} />
        </TouchableOpacity>

        <View style={styles.errorContainer}>
          <Ionicons name="warning-outline" size={50} color="#FF3B30" />
          <Text style={[dynamicStyles.text, styles.errorText]}>{error}</Text>
          <TouchableOpacity 
            style={styles.retryButton} 
            onPress={fetchReportes}
          >
            <Text style={styles.retryButtonText}>Reintentar</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={[{ flex: 1 }, dynamicStyles.container]}>
      {/* Botón de volver */}
      <TouchableOpacity
        onPress={() => router.replace('/elegir')}
        style={[styles.backButton, temaOscuro ? styles.backDark : styles.backLight]}
      >
        <Ionicons name="arrow-back" size={24} color={temaOscuro ? '#fff' : '#000'} />
      </TouchableOpacity>

      <ScrollView 
        contentContainerStyle={styles.body}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={onRefresh}
            colors={['#4a90e2']}
            tintColor="#4a90e2"
          />
        }
      >
        <Text style={[styles.header, dynamicStyles.text]}>Registro de Reportes</Text>

        {reportes.length > 0 ? (
          reportes.map((reporte) => (
            <View key={reporte.id} style={[styles.card, dynamicStyles.card]}>
              <View style={styles.row}>
                <Ionicons name="document-text-outline" size={22} color="#4a90e2" />
                <Text style={[styles.label, dynamicStyles.text]}>{reporte.titulo}</Text>
              </View>

              <View style={styles.row}>
                <Ionicons name="person-circle-outline" size={18} color="#4a90e2" />
                <Text style={[dynamicStyles.subtext, { marginLeft: 5 }]}>{reporte.nombre}</Text>
              </View>

              <View style={[styles.row, { marginTop: 8, justifyContent: 'space-between' }]}>
                <Text style={[styles.sub, dynamicStyles.subtext]}>
                  {reporte.fecha} – {reporte.hora}
                </Text>
                <Text style={[styles.sub, dynamicStyles.subtext]}>
                  {reporte.ubicacion}
                </Text>
              </View>

              <View style={[styles.row, { marginTop: 8 }]}>
                <Text style={{ ...styles.estado, ...getEstadoColorEstilo(reporte.tipo) }}>
                  {reporte.tipo}
                </Text>
                <Text style={[styles.resultado, getSeveridadColor(reporte.severidad)]}>
                  {reporte.severidad}
                </Text>
              </View>

              <Text style={[styles.reporteTexto, dynamicStyles.subtext, { marginTop: 10 }]}>
                {reporte.descripcion}
              </Text>

              {reporte.evidencias.length > 0 && (
                <View style={{ marginTop: 10 }}>
                  <Text style={[dynamicStyles.text, { fontSize: 14, marginBottom: 5 }]}>
                    Evidencias:
                  </Text>
                  <ScrollView horizontal showsHorizontalScrollIndicator={false}>
                    {reporte.evidencias.map((evidencia, index) => (
                      <TouchableOpacity 
                        key={index} 
                        style={styles.evidenceContainer}
                        onPress={() => handleOpenEvidence(evidencia)}
                      >
                        <Image
                          source={{ uri: evidencia }}
                          style={styles.evidenceImage}
                          resizeMode="cover"
                        />
                      </TouchableOpacity>
                    ))}
                  </ScrollView>
                </View>
              )}
            </View>
          ))
        ) : (
          <View style={styles.emptyContainer}>
            <Ionicons name="document-outline" size={50} color="#aaa" />
            <Text style={[dynamicStyles.text, styles.emptyText]}>No hay reportes disponibles</Text>
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}


// Colores para tipo de reporte
function getEstadoColorEstilo(tipo) {
  switch (tipo) {
    case 'Error del sistema':
      return { color: '#b22921', fontWeight: '600' };
    case 'Fallo autenticación':
      return { color: '#FF9500', fontWeight: '600' };
    case 'Acceso no autorizado':
      return { color: '#FF3B30', fontWeight: '600' };
    case 'Horario irregular':
      return { color: '#FFCC00', fontWeight: '600' };
    default:
      return { color: '#007AFF', fontWeight: '600' };
  }
}


// Colores para severidad
function getSeveridadColor(severidad) {
  switch (severidad) {
    case 'Crítica':
      return { color: '#FF3B30', fontWeight: '600' };
    case 'Alta':
      return { color: '#FF9500', fontWeight: '600' };
    case 'Media':
      return { color: '#FFCC00', fontWeight: '600' };
    case 'Baja':
      return { color: '#34C759', fontWeight: '600' };
    default:
      return { color: '#555', fontWeight: '600' };
  }
}

const styles = StyleSheet.create({
  backButton: {
    position: 'absolute',
    top: 40,
    left: 20,
    padding: 10,
    borderRadius: 50,
    zIndex: 10,
  },
  backLight: {
    backgroundColor: 'rgba(0,0,0,0.05)',
  },
  backDark: {
    backgroundColor: 'rgba(255,255,255,0.1)',
  },
  body: {
    paddingTop: 100,
    paddingBottom: 50,
    paddingHorizontal: 24,
  },
  header: {
    fontSize: 24,
    fontWeight: '800',
    marginBottom: 30,
    textAlign: 'center',
  },
  card: {
    borderRadius: 16,
    padding: 20,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 3 },
    shadowOpacity: 0.1,
    shadowRadius: 6,
    elevation: 3,
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    marginBottom: 4,
  },
  label: {
    fontSize: 16,
    fontWeight: '600',
  },
  sub: {
    fontSize: 14,
  },
  estado: {
    fontSize: 14,
    marginRight: 15,
  },
  resultado: {
    fontSize: 14,
  },
  reporteTexto: {
    fontSize: 14,
    lineHeight: 20,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  errorText: {
    marginTop: 15,
    textAlign: 'center',
    fontSize: 16,
  },
  retryButton: {
    marginTop: 20,
    backgroundColor: '#4a90e2',
    paddingVertical: 10,
    paddingHorizontal: 20,
    borderRadius: 8,
  },
  retryButtonText: {
    color: 'white',
    fontWeight: '600',
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    marginTop: 100,
  },
  emptyText: {
    marginTop: 15,
    fontSize: 16,
    color: '#aaa',
  },
  evidenceContainer: {
    width: 100,
    height: 100,
    borderRadius: 8,
    overflow: 'hidden',
    marginRight: 10,
    backgroundColor: '#f0f0f0',
  },
  evidenceImage: {
    width: '100%',
    height: '100%',
  },
});

const lightStyles = StyleSheet.create({
  container: {
    backgroundColor: '#f7f9fc',
  },
  text: {
    color: '#111',
  },
  subtext: {
    color: '#555',
  },
  card: {
    backgroundColor: '#fff',
  },
});

const darkStyles = StyleSheet.create({
  container: {
    backgroundColor: '#1C1C1E',
  },
  text: {
    color: '#fff',
  },
  subtext: {
    color: '#aaa',
  },
  card: {
    backgroundColor: '#2C2C2E',
  },
});