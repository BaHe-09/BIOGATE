import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { useEffect, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  FlatList,
  Image,
  RefreshControl,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import { useTheme } from '../context/ThemeContext';
import BottomNavBar from './BottomNavBar';

interface HistorialItem {
  id_acceso: number;
  nombre_completo: string;
  fecha: string;
  resultado: string;
  dispositivo: string;
  foto_url?: string;
}

export default function HomeScreen() {
  const router = useRouter();
  const { temaOscuro } = useTheme();
  const dynamicStyles = temaOscuro ? darkStyles : lightStyles;
  
  const [historial, setHistorial] = useState<HistorialItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  const fetchHistorial = async () => {
    try {
      setLoading(true);
      // Reemplaza esta URL con tu endpoint real
      const response = await fetch('https://render-biogate.onrender.com/historial-accesos/');
      
      if (!response.ok) {
        throw new Error(`Error HTTP: ${response.status}`);
      }
      
      const data = await response.json();
      setHistorial(data);
    } catch (error) {
      console.error('Error fetching historial:', error);
      Alert.alert(
        'Error',
        'No se pudo cargar el historial de accesos. Por favor, inténtalo de nuevo más tarde.',
        [{ text: 'OK' }]
      );
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchHistorial();
  }, []);

  const onRefresh = () => {
    setRefreshing(true);
    fetchHistorial();
  };

  const filteredHistorial = historial.filter(item =>
    item.nombre_completo.toLowerCase().includes(searchQuery.toLowerCase()) ||
    item.dispositivo.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const renderItem = ({ item }: { item: HistorialItem }) => (
    <TouchableOpacity
      onPress={() =>
        router.push({
          pathname: '/usuario-detalle',
          params: {
            id: item.id_acceso.toString(),
            name: item.nombre_completo,
            time: item.fecha,
            status: item.resultado,
            location: item.dispositivo,
            photoUrl: item.foto_url || ''
          }
        })
      }
    >
      <View style={styles.item}>
        <View style={[styles.avatarContainer, dynamicStyles.avatarContainer]}>
          {item.foto_url ? (
            <Image
              source={{ uri: item.foto_url }}
              style={styles.avatar}
            />
          ) : (
            <Ionicons 
              name="person-circle-outline" 
              size={50} 
              color={temaOscuro ? '#777' : '#999'} 
            />
          )}
        </View>
        <View style={{ flex: 1 }}>
          <Text 
            style={[
              styles.name, 
              dynamicStyles.text,
              item.nombre_completo === 'DESCONOCIDO' && styles.unknownName
            ]}
          >
            {item.nombre_completo}
            {item.nombre_completo === 'DESCONOCIDO' && (
              <Text style={styles.unknownBadge}> • No identificado</Text>
            )}
          </Text>
          <Text style={[styles.time, dynamicStyles.subtext]}>
            {item.fecha} –{' '}
            <Text style={item.resultado === 'PERMITIDO' ? styles.allowed : styles.denied}>
              {item.resultado}
            </Text>
          </Text>
          <Text style={[styles.device, dynamicStyles.subtext]}>
            <Ionicons name="location-outline" size={14} /> {item.dispositivo}
          </Text>
        </View>
        {item.resultado === 'DENEGADO' && <View style={styles.dot} />}
        <Ionicons 
          name="chevron-forward" 
          size={20} 
          color={temaOscuro ? '#ccc' : '#888'} 
        />
      </View>
    </TouchableOpacity>
  );

  return (
    <View style={[styles.container, dynamicStyles.container]}>
      {/* Header */}
      <View style={styles.topHeader}>
        <View style={styles.titleContainer}>
          <Text style={[styles.title, dynamicStyles.text]}>Historial de accesos</Text>
        </View>
      </View>

      {/* Barra de búsqueda */}
      <View style={[styles.searchContainer, dynamicStyles.searchContainer]}>
        <Ionicons 
          name="search" 
          size={20} 
          color={temaOscuro ? '#ccc' : '#aaa'} 
          style={styles.searchIcon} 
        />
        <TextInput
          placeholder="Buscar por nombre o dispositivo..."
          placeholderTextColor={temaOscuro ? '#ccc' : '#aaa'}
          style={[styles.searchInput, dynamicStyles.text]}
          value={searchQuery}
          onChangeText={setSearchQuery}
          clearButtonMode="while-editing"
        />
      </View>

      {/* Contenido principal */}
      {loading ? (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={temaOscuro ? '#0A84FF' : '#007AFF'} />
        </View>
      ) : (
        <FlatList
          data={filteredHistorial}
          keyExtractor={(item) => item.id_acceso.toString()}
          contentContainerStyle={{ paddingBottom: 100 }}
          refreshControl={
            <RefreshControl
              refreshing={refreshing}
              onRefresh={onRefresh}
              colors={[temaOscuro ? '#0A84FF' : '#007AFF']}
              tintColor={temaOscuro ? '#0A84FF' : '#007AFF'}
            />
          }
          renderItem={renderItem}
          ListEmptyComponent={
            <View style={styles.emptyContainer}>
              <Ionicons 
                name="time-outline" 
                size={60} 
                color={temaOscuro ? '#555' : '#ccc'} 
              />
              <Text style={[styles.emptyText, dynamicStyles.text]}>
                {searchQuery 
                  ? 'No se encontraron resultados' 
                  : 'No hay accesos registrados'}
              </Text>
            </View>
          }
        />
      )}

      <BottomNavBar />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    paddingHorizontal: 16,
    paddingTop: 50,
  },
  topHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 20,
  },
  editButton: {
    width: 40,
    height: 40,
    justifyContent: 'center',
    alignItems: 'center',
  },
  filterButton: {
    width: 40,
    height: 40,
    justifyContent: 'center',
    alignItems: 'center',
  },
  titleContainer: {
    flex: 1,
    alignItems: 'center',
  },
  title: {
    fontSize: 20,
    fontWeight: '700',
  },
  searchContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    borderRadius: 10,
    paddingHorizontal: 15,
    marginBottom: 15,
    height: 45,
  },
  searchIcon: {
    marginRight: 10,
  },
  searchInput: {
    flex: 1,
    fontSize: 16,
  },
  item: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 15,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: '#ddd',
  },
  avatarContainer: {
    width: 50,
    height: 50,
    marginRight: 15,
    justifyContent: 'center',
    alignItems: 'center',
    borderRadius: 25,
  },
  avatar: {
    width: 50,
    height: 50,
    borderRadius: 25,
  },
  name: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 3,
  },
  time: {
    fontSize: 13,
    marginBottom: 3,
  },
  device: {
    fontSize: 13,
    opacity: 0.8,
  },
  allowed: {
    color: 'green',
    fontWeight: 'bold',
  },
  denied: {
    color: 'red',
    fontWeight: 'bold',
  },
  dot: {
    width: 10,
    height: 10,
    backgroundColor: '#007AFF',
    borderRadius: 5,
    marginRight: 10,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
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
    opacity: 0.6,
  },
  unknownName: {
    color: '#888',
  },
  unknownBadge: {
    fontSize: 12,
    color: '#FF9500',
    fontStyle: 'italic',
  },
});

const lightStyles = StyleSheet.create({
  container: { backgroundColor: '#fff' },
  text: { color: '#222' },
  subtext: { color: '#555' },
  link: { color: '#007AFF' },
  searchContainer: { backgroundColor: '#F2F2F2' },
  avatarContainer: { backgroundColor: '#F2F2F2' },
  unknownName: { color: '#888' },
  unknownBadge: { color: '#FF9500' },
});

const darkStyles = StyleSheet.create({
  container: { backgroundColor: '#1C1C1E' },
  text: { color: '#fff' },
  subtext: { color: '#ccc' },
  link: { color: '#0A84FF' },
  searchContainer: { backgroundColor: '#2C2C2E' },
  avatarContainer: { backgroundColor: '#2C2C2E' },
  unknownName: { color: '#777' },
  unknownBadge: { color: '#FF9F0A' },
});