import { Ionicons } from '@expo/vector-icons';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { useEffect, useState } from 'react';
import { ActivityIndicator, Image, ScrollView, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { useTheme } from '../context/ThemeContext';

interface Dispositivo {
  nombre: string;
  ubicacion: string;
}

interface DetallesAcceso {
  hora_entrada: string;
  hora_salida: string;
}

interface DetalleAcceso {
  id_acceso: number;
  nombre_completo: string;
  fecha: string;
  horario: string;
  dispositivo: Dispositivo;
  estatus: string;
  nivel_confianza?: number;
  razon: string;
  detalles_acceso: DetallesAcceso;
  es_dia_laboral: boolean;
  estado_registro: string;
  dias_laborales?: string;
  foto_url?: string;
}

export default function UsuarioDetalle() {
  const router = useRouter();
  const { temaOscuro } = useTheme();
  const { id } = useLocalSearchParams();
  const dynamicStyles = temaOscuro ? darkStyles : lightStyles;
  const [detalle, setDetalle] = useState<DetalleAcceso | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchDetalle = async () => {
      try {
        const response = await fetch(`https://render-biogate.onrender.com/historial-accesos/${id}`);
        const data = await response.json();
        setDetalle(data);
      } catch (error) {
        console.error('Error fetching detalle:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchDetalle();
  }, [id]);

  const goBack = () => {
    router.back();
  };

  const getStatusColor = () => {
    return detalle?.estatus === 'PERMITIDO' ? '#4CAF50' : '#F44336';
  };

  const getStatusIcon = () => {
    return detalle?.estatus === 'PERMITIDO' ? 'checkmark-circle' : 'close-circle';
  };

  const getEstadoRegistroColor = () => {
    switch(detalle?.estado_registro) {
      case 'ENTRADA': return '#4CAF50';
      case 'RETRASO': return '#FFC107';
      case 'SALIDA': return '#2196F3';
      case 'HORAS_EXTRAS': return '#9C27B0';
      default: return dynamicStyles.subtext.color;
    }
  };

  const getDiaLaboralColor = () => {
    return detalle?.es_dia_laboral ? '#4CAF50' : '#F44336';
  };

  const formatDiasLaborales = (dias?: string) => {
    if (!dias) return 'N/A';
    const diasMap: Record<string, string> = {
      'L-V': 'Lunes a Viernes',
      'L-S': 'Lunes a Sábado',
      'L-D': 'Todos los días'
    };
    return diasMap[dias] || dias;
  };

  if (loading) {
    return (
      <View style={[styles.loadingContainer, dynamicStyles.container]}>
        <ActivityIndicator size="large" color={temaOscuro ? '#0A84FF' : '#007AFF'} />
      </View>
    );
  }

  if (!detalle) {
    return (
      <View style={[styles.container, dynamicStyles.container]}>
        <Text style={[styles.errorText, dynamicStyles.text]}>No se encontraron detalles del acceso</Text>
        <TouchableOpacity style={styles.button} onPress={goBack}>
          <Text style={styles.buttonText}>Volver</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <ScrollView 
      style={[styles.container, dynamicStyles.container]}
      contentContainerStyle={styles.scrollContent}
    >
      {/* Botón de cerrar */}
      <TouchableOpacity 
        onPress={goBack}
        style={[
          styles.closeButton,
          temaOscuro ? styles.closeButtonDark : styles.closeButtonLight
        ]}
      >
        <Ionicons 
          name="close" 
          size={24} 
          color={temaOscuro ? '#fff' : '#000'} 
        />
      </TouchableOpacity>

      <Text style={[styles.title, dynamicStyles.text]}>Detalles del acceso</Text>

      <View style={[styles.card, temaOscuro ? styles.cardDark : styles.cardLight]}>
        {detalle.foto_url && (
          <Image
            source={{ uri: detalle.foto_url }}
            style={styles.avatar}
          />
        )}

        <Text style={[styles.name, dynamicStyles.text]}>
          {detalle.nombre_completo === 'DESCONOCIDO' ? 'Persona no identificada' : detalle.nombre_completo}
        </Text>

        {/* Información básica */}
        <View style={styles.section}>
          <Text style={[styles.sectionTitle, dynamicStyles.text]}>Información del acceso</Text>
          
          <View style={styles.infoGroup}>
            <Ionicons name="calendar-outline" size={20} color={dynamicStyles.subtext.color} />
            <Text style={[styles.label, dynamicStyles.subtext]}>Fecha: {detalle.fecha}</Text>
          </View>

          <View style={styles.infoGroup}>
            <Ionicons name="time-outline" size={20} color={dynamicStyles.subtext.color} />
            <Text style={[styles.label, dynamicStyles.subtext]}>Horario: {detalle.horario}</Text>
          </View>

          <View style={styles.infoGroup}>
            <Ionicons name="hardware-chip-outline" size={20} color={dynamicStyles.subtext.color} />
            <Text style={[styles.label, dynamicStyles.subtext]}>
              Dispositivo: {detalle.dispositivo.nombre}
            </Text>
          </View>

          <View style={styles.infoGroup}>
            <Ionicons name="location-outline" size={20} color={dynamicStyles.subtext.color} />
            <Text style={[styles.label, dynamicStyles.subtext]}>
              Ubicación: {detalle.dispositivo.ubicacion}
            </Text>
          </View>

          <View style={styles.infoGroup}>
            <Ionicons
              name={getStatusIcon()}
              size={20}
              color={getStatusColor()}
            />
            <Text style={[styles.label, { color: getStatusColor() }]}>
              Estatus: {detalle.estatus}
            </Text>
          </View>

          {detalle.nivel_confianza && (
            <View style={styles.infoGroup}>
              <Ionicons name="stats-chart-outline" size={20} color={dynamicStyles.subtext.color} />
              <Text style={[styles.label, dynamicStyles.subtext]}>
                Nivel de confianza: {detalle.nivel_confianza.toFixed(2)}%
              </Text>
            </View>
          )}

          <View style={styles.infoGroup}>
            <Ionicons name="information-circle-outline" size={20} color={dynamicStyles.subtext.color} />
            <Text style={[styles.label, dynamicStyles.subtext]}>
              Razón: {detalle.razon}
            </Text>
          </View>
        </View>

        {/* Detalles de horario */}
        <View style={styles.section}>
          <Text style={[styles.sectionTitle, dynamicStyles.text]}>Horario</Text>
          
          <View style={styles.infoGroup}>
            <Ionicons name="log-in-outline" size={20} color={dynamicStyles.subtext.color} />
            <Text style={[styles.label, dynamicStyles.subtext]}>
              Hora entrada: {detalle.detalles_acceso.hora_entrada || 'N/A'}
            </Text>
          </View>

          <View style={styles.infoGroup}>
            <Ionicons name="log-out-outline" size={20} color={dynamicStyles.subtext.color} />
            <Text style={[styles.label, dynamicStyles.subtext]}>
              Hora salida: {detalle.detalles_acceso.hora_salida || 'N/A'}
            </Text>
          </View>

          <View style={styles.infoGroup}>
            <Ionicons name="calendar-outline" size={20} color={dynamicStyles.subtext.color} />
            <Text style={[styles.label, dynamicStyles.subtext]}>
              Días registrados: {formatDiasLaborales(detalle.dias_laborales)}
            </Text>
          </View>

          <View style={styles.infoGroup}>
            <Ionicons 
              name={detalle.es_dia_laboral ? 'checkmark-circle-outline' : 'close-circle-outline'} 
              size={20} 
              color={getDiaLaboralColor()} 
            />
            <Text style={[styles.label, { color: getDiaLaboralColor() }]}>
              Día laboral: {detalle.es_dia_laboral ? 'Sí' : 'No'}
            </Text>
          </View>

          <View style={styles.infoGroup}>
            <Ionicons name="pricetag-outline" size={20} color={getEstadoRegistroColor()} />
            <Text style={[styles.label, { color: getEstadoRegistroColor() }]}>
              Estado registro: {detalle.estado_registro || 'N/A'}
            </Text>
          </View>
        </View>
      </View>

      <TouchableOpacity 
        style={[styles.button, { backgroundColor: getStatusColor() }]} 
        onPress={goBack}
      >
        <Text style={styles.buttonText}>Volver al historial</Text>
      </TouchableOpacity>
    </ScrollView>
  );
}

// Estilos
const styles = StyleSheet.create({
  container: {
    flex: 1,
    paddingHorizontal: 20,
  },
  scrollContent: {
    paddingTop: 70,
    paddingBottom: 40,
    alignItems: 'center',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  errorText: {
    fontSize: 18,
    marginBottom: 20,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 25,
    textAlign: 'center',
  },
  card: {
    width: '100%',
    padding: 25,
    borderRadius: 15,
    marginBottom: 20,
    shadowOffset: { width: 0, height: 4 },
    shadowRadius: 8,
    elevation: 5,
  },
  cardLight: {
    backgroundColor: '#f8f8f8',
    shadowColor: '#000',
    shadowOpacity: 0.1,
  },
  cardDark: {
    backgroundColor: '#2c2c2e',
    shadowColor: '#fff',
    shadowOpacity: 0.05,
  },
  avatar: {
    width: 120,
    height: 120,
    borderRadius: 60,
    marginBottom: 20,
    alignSelf: 'center',
    backgroundColor: '#f0f0f0',
  },
  name: {
    fontSize: 22,
    fontWeight: '600',
    marginBottom: 25,
    textAlign: 'center',
  },
  section: {
    marginBottom: 25,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 15,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: '#ddd',
    paddingBottom: 8,
  },
  infoGroup: {
    flexDirection: 'row',
    alignItems: 'center',
    marginVertical: 10,
  },
  label: {
    fontSize: 16,
    marginLeft: 8,
    flexShrink: 1,
  },
  button: {
    width: '80%',
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  closeButton: {
    position: 'absolute',
    top: 50,
    right: 20,
    width: 40,
    height: 40,
    borderRadius: 20,
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 10,
  },
  closeButtonLight: {
    backgroundColor: 'rgba(0,0,0,0.1)',
  },
  closeButtonDark: {
    backgroundColor: 'rgba(255,255,255,0.1)',
  },
});

const lightStyles = StyleSheet.create({
  container: {
    backgroundColor: '#fff',
  },
  text: {
    color: '#222',
  },
  subtext: {
    color: '#555',
  },
});

const darkStyles = StyleSheet.create({
  container: {
    backgroundColor: '#121212',
  },
  text: {
    color: '#fff',
  },
  subtext: {
    color: '#ccc',
  },
});