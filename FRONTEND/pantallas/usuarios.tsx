import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { useEffect, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  FlatList,
  RefreshControl,
  SafeAreaView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import { useTheme } from '../context/ThemeContext';
import BottomNavBar from './BottomNavBar';

interface Usuario {
  id_persona: number;
  nombre: string;
  apellido_paterno: string;
  apellido_materno?: string;
  correo_electronico?: string;
  telefono?: string;
  activo: boolean;
  id_rol: number;
  nombre_rol: string;
  es_admin: boolean;
  foto_url?: string;
}

export default function UsuariosScreen() {
  const router = useRouter();
  const { temaOscuro } = useTheme();
  const dynamicStyles = temaOscuro ? darkStyles : lightStyles;

  const [usuarios, setUsuarios] = useState<Usuario[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [search, setSearch] = useState('');

  const fetchUsuarios = async () => {
    try {
      setLoading(true);
      const response = await fetch('https://render-biogate.onrender.com/personas/');
      const data = await response.json();

      const usuariosMapeados = data.map((persona: any) => ({
        id_persona: persona.id_persona,
        nombre: persona.nombre,
        apellido_paterno: persona.apellido_paterno,
        apellido_materno: persona.apellido_materno,
        correo_electronico: persona.correo_electronico,
        telefono: persona.telefono,
        activo: persona.activo,
        id_rol: persona.id_rol || 2,
        nombre_rol: persona.nombre_rol || 'Usuario',
        es_admin: persona.id_rol === 1,
        foto_url: persona.foto_url
      }));

      setUsuarios(usuariosMapeados);
    } catch (error) {
      console.error('Error fetching usuarios:', error);
      Alert.alert('Error', 'No se pudieron cargar los usuarios');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchUsuarios();
  }, []);

  const onRefresh = () => {
    setRefreshing(true);
    fetchUsuarios();
  };

  const actualizarEstadoUsuario = async (id_persona: number, nuevoEstado: boolean) => {
    try {
      const response = await fetch(`https://render-biogate.onrender.com/personas/${id_persona}/estado`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ activo: nuevoEstado }),
      });

      if (response.ok) {
        setUsuarios(usuarios.map(usuario => 
          usuario.id_persona === id_persona 
            ? { ...usuario, activo: nuevoEstado } 
            : usuario
        ));
      } else {
        throw new Error('Error al actualizar');
      }
    } catch (error) {
      console.error('Error actualizando estado:', error);
      Alert.alert('Error', 'No se pudo actualizar el estado del usuario');
      fetchUsuarios();
    }
  };

  const eliminarUsuario = async (id_persona: number) => {
    try {
      Alert.alert(
        "Confirmar eliminación",
        "¿Estás seguro de que deseas eliminar este usuario y todos sus datos asociados? Esta acción no se puede deshacer.",
        [
          {
            text: "Cancelar",
            style: "cancel"
          },
          { 
            text: "Eliminar", 
            onPress: async () => {
              const response = await fetch(`https://render-biogate.onrender.com/personas/${id_persona}`, {
                method: 'DELETE',
              });

              if (response.ok) {
                setUsuarios(usuarios.filter(usuario => usuario.id_persona !== id_persona));
                Alert.alert("Éxito", "Usuario eliminado correctamente");
              } else {
                throw new Error('Error al eliminar');
              }
            },
            style: "destructive"
          }
        ]
      );
    } catch (error) {
      console.error('Error eliminando usuario:', error);
      Alert.alert('Error', 'No se pudo eliminar el usuario');
    }
  };

  const getNombreCompleto = (usuario: Usuario) => {
    return `${usuario.nombre} ${usuario.apellido_paterno}${usuario.apellido_materno ? ` ${usuario.apellido_materno}` : ''}`;
  };

 

  const filteredUsuarios = usuarios.filter(usuario => {
    const nombreCompleto = getNombreCompleto(usuario).toLowerCase();
    const searchTerm = search.toLowerCase();
    return nombreCompleto.includes(searchTerm) ||
      (usuario.correo_electronico && usuario.correo_electronico.toLowerCase().includes(searchTerm));
  });

  const renderUsuario = ({ item }: { item: Usuario }) => (
    <View style={[styles.usuarioCard, dynamicStyles.card]}>
      <View style={styles.cardContent}>
        <View style={styles.userInfoRow}>
          <View style={styles.avatarContainer}>
            <Ionicons 
              name="person-circle-outline" 
              size={50} 
              color={temaOscuro ? '#ccc' : '#888'} 
            />
          </View>
          <View style={styles.usuarioInfo}>
            <View style={styles.nombreContainer}>
              <Text style={[styles.usuarioNombre, dynamicStyles.text]} numberOfLines={1}>
                {getNombreCompleto(item)}
              </Text>
              {item.es_admin && (
                <View style={styles.adminBadge}>
                  <Ionicons name="shield-checkmark" size={14} color="#4CAF50" />
                  <Text style={styles.adminText}>Admin</Text>
                </View>
              )}
            </View>
            {item.correo_electronico && (
              <Text style={[styles.usuarioEmail, dynamicStyles.subtext]} numberOfLines={1}>
                <Ionicons name="mail-outline" size={12} color={dynamicStyles.subtext.color} /> {item.correo_electronico}
              </Text>
            )}
            <View style={styles.detallesContainer}>
              <Text style={[styles.usuarioRol, dynamicStyles.subtext]}>
                <Ionicons name="person-outline" size={12} color={dynamicStyles.subtext.color} /> {item.nombre_rol}
              </Text>
              {item.telefono && (
                <Text style={[styles.usuarioTelefono, dynamicStyles.subtext]}>
                  <Ionicons name="call-outline" size={12} color={dynamicStyles.subtext.color} /> {item.telefono}
                </Text>
              )}
            </View>
          </View>
        </View>
        {!item.es_admin ? (
          <View style={styles.accessButtons}>
            <TouchableOpacity
              style={[
                styles.accessBtn,
                item.activo && styles.accessSelected,
                { borderColor: temaOscuro ? '#0A84FF' : '#007AFF' }
              ]}
              onPress={() => actualizarEstadoUsuario(item.id_persona, true)}
            >
              <Text style={[
                styles.accessText,
                item.activo && styles.accessTextSelected,
              ]}>
                PERMITIDO
              </Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[
                styles.accessBtn,
                !item.activo && styles.accessSelected,
                { borderColor: temaOscuro ? '#0A84FF' : '#007AFF' }
              ]}
              onPress={() => actualizarEstadoUsuario(item.id_persona, false)}
            >
              <Text style={[
                styles.accessText,
                !item.activo && styles.accessTextSelected,
              ]}>
                DENEGADO
              </Text>
            </TouchableOpacity>
          </View>
        ) : (
          <View style={[styles.statusIndicator, { backgroundColor: '#4CAF50' }]}>
            <Text style={styles.statusIndicatorText}>ACTIVO</Text>
          </View>
        )}
        
        {/* Botón de eliminar */}
        {!item.es_admin && (
          <TouchableOpacity
            style={[styles.deleteButton, { backgroundColor: temaOscuro ? '#3A3A3C' : '#F5F5F5' }]}
            onPress={() => eliminarUsuario(item.id_persona)}
          >
            <Ionicons name="trash-outline" size={20} color="#FF3B30" />
          </TouchableOpacity>
        )}
      </View>
    </View>
  );

  return (
    <SafeAreaView style={[styles.container, dynamicStyles.container]}>
      {/* Header */}
      <View style={styles.header}>
        <View style={[styles.searchContainer, dynamicStyles.searchContainer]}>
          <Ionicons 
            name="search" 
            size={20} 
            color={temaOscuro ? '#ccc' : '#aaa'} 
            style={styles.searchIcon} 
          />
          <TextInput
            placeholder="Buscar por nombre o correo..."
            placeholderTextColor={temaOscuro ? '#aaa' : '#666'}
            value={search}
            onChangeText={setSearch}
            style={[styles.searchInput, dynamicStyles.text]}
            clearButtonMode="while-editing"
          />
        </View>

        <TouchableOpacity 
          onPress={() => router.push('/nuevo-usuario')} 
          style={[styles.addButton, dynamicStyles.addButton]}
        >
          <Ionicons name="add" size={26} color={temaOscuro ? '#0A84FF' : '#007AFF'} />
        </TouchableOpacity>
      </View>

      {/* Contenido */}
      {loading ? (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={temaOscuro ? '#0A84FF' : '#007AFF'} />
        </View>
      ) : (
        <FlatList
          data={filteredUsuarios}
          renderItem={renderUsuario}
          keyExtractor={(item) => item.id_persona.toString()}
          contentContainerStyle={styles.listContent}
          refreshControl={
            <RefreshControl
              refreshing={refreshing}
              onRefresh={onRefresh}
              colors={[temaOscuro ? '#0A84FF' : '#007AFF']}
              tintColor={temaOscuro ? '#0A84FF' : '#007AFF'}
            />
          }
          ListEmptyComponent={
            <View style={styles.emptyContainer}>
              <Ionicons 
                name="people-outline" 
                size={50} 
                color={temaOscuro ? '#555' : '#ccc'} 
              />
              <Text style={[styles.emptyText, dynamicStyles.text]}>
                {search ? 'No se encontraron usuarios' : 'No hay usuarios registrados'}
              </Text>
            </View>
          }
        />
      )}

      {/* Bottom Navigation Bar */}
      <BottomNavBar />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    paddingBottom: 60, // Espacio para el BottomNavBar
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingTop: 50,
    paddingBottom: 8,
    marginBottom: 8,
  },
  searchContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    borderRadius: 10,
    flex: 1,
    paddingHorizontal: 12,
    height: 40,
  },
  searchIcon: {
    marginRight: 8,
  },
  searchInput: {
    flex: 1,
    fontSize: 16,
    paddingVertical: 8,
  },
  addButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    justifyContent: 'center',
    alignItems: 'center',
    marginLeft: 8,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  listContent: {
    paddingHorizontal: 16,
    paddingBottom: 16,
  },
  usuarioCard: {
    borderRadius: 12,
    marginBottom: 12,
    shadowOffset: { width: 0, height: 1 },
    shadowRadius: 3,
    elevation: 2,
  },
  cardContent: {
    padding: 16,
  },
  userInfoRow: {
    flexDirection: 'row',
    marginBottom: 12,
  },
  avatarContainer: {
    width: 50,
    height: 50,
    borderRadius: 25,
    marginRight: 16,
    justifyContent: 'center',
    alignItems: 'center',
  },
  usuarioInfo: {
    flex: 1,
  },
  nombreContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 4,
  },
  usuarioNombre: {
    fontSize: 16,
    fontWeight: '600',
    flexShrink: 1,
    marginRight: 8,
  },
  usuarioEmail: {
    fontSize: 13,
    marginBottom: 4,
    opacity: 0.9,
  },
  detallesContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    flexWrap: 'wrap',
  },
  usuarioRol: {
    fontSize: 12,
    marginRight: 12,
    opacity: 0.8,
  },
  usuarioTelefono: {
    fontSize: 12,
    opacity: 0.8,
  },
  accessButtons: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginTop: 8,
    paddingTop: 8,
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: '#ddd',
  },
  accessBtn: {
    borderWidth: 1,
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderRadius: 15,
    minWidth: 120,
    alignItems: 'center',
    justifyContent: 'center',
  },
  accessText: {
    fontSize: 14,
    fontWeight: '600',
  },
  accessSelected: {
    backgroundColor: '#007AFF',
  },
  accessTextSelected: {
    color: '#fff',
  },
  adminBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(76, 175, 80, 0.1)',
    paddingVertical: 2,
    paddingHorizontal: 6,
    borderRadius: 8,
  },
  adminText: {
    color: '#4CAF50',
    fontSize: 12,
    fontWeight: '600',
    marginLeft: 4,
  },
  statusIndicator: {
    paddingVertical: 8,
    borderRadius: 15,
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 8,
  },
  statusIndicatorText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    marginTop: 100,
  },
  emptyText: {
    marginTop: 16,
    fontSize: 16,
    textAlign: 'center',
    opacity: 0.6,
  },
  deleteButton: {
    position: 'absolute',
    top: 8,
    right: 8,
    width: 32,
    height: 32,
    borderRadius: 16,
    justifyContent: 'center',
    alignItems: 'center',
  },
});

const lightStyles = StyleSheet.create({
  container: {
    backgroundColor: '#F9FAFB',
  },
  text: {
    color: '#222',
  },
  subtext: {
    color: '#555',
  },
  searchContainer: {
    backgroundColor: '#EFEFEF',
  },
  card: {
    backgroundColor: '#fff',
    shadowColor: '#000',
    shadowOpacity: 0.05,
  },
  addButton: {
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
    color: '#ccc',
  },
  searchContainer: {
    backgroundColor: '#2C2C2E',
  },
  card: {
    backgroundColor: '#2C2C2E',
    shadowColor: '#fff',
    shadowOpacity: 0.02,
  },
  addButton: {
    backgroundColor: '#2C2C2E',
  },
});