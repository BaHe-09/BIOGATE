import { Ionicons } from '@expo/vector-icons';
import * as FileSystem from 'expo-file-system';
import { useRouter } from 'expo-router';
import * as Sharing from 'expo-sharing';
import { Alert, Pressable, StyleSheet, Text, View } from 'react-native';
import BottomNavBar from '../app/BottomNavBar';
import { useTheme } from '../context/ThemeContext';

export default function AdministracionScreen() {
  const router = useRouter();
  const { temaOscuro } = useTheme();
  const dynamicStyles = temaOscuro ? darkStyles : lightStyles;
  
  // Reemplaza con la URL base de tu API
  const API_BASE_URL = 'https://render-biogate-990i.onrender.com'; 

  // Función para manejar la descarga de archivos
const downloadFile = async (endpoint: string, filename: string) => {
  try {
    Alert.alert('Descargando', `Preparando ${filename}...`);
    
    const response = await fetch(`${API_BASE_URL}/${endpoint}`);
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(errorText || 'Error al descargar el archivo');
    }
    
    const csvData = await response.text();
    const fileUri = `${FileSystem.documentDirectory}${filename}`;
    
    await FileSystem.writeAsStringAsync(fileUri, csvData, {
      encoding: FileSystem.EncodingType.UTF8,
    });
    
    if (await Sharing.isAvailableAsync()) {
      await Sharing.shareAsync(fileUri, {
        mimeType: 'text/csv',
        dialogTitle: `Descargar ${filename}`,
        UTI: 'public.comma-separated-values-text',
      });
    } else {
      Alert.alert(
        'Descarga completada',
        `El archivo se ha guardado en: ${fileUri}`,
        [{ text: 'OK', onPress: () => console.log('Descarga finalizada') }]
      );
    }
    
  } catch (error) {
    console.error('Error en la descarga:', error);
    
    let errorMessage = 'Error desconocido al descargar';
    
    // Verificar si es un Error estándar
    if (error instanceof Error) {
      errorMessage = error.message;
    } 
    // Verificar si es una cadena (poco común, pero posible)
    else if (typeof error === 'string') {
      errorMessage = error;
    }
    
    Alert.alert('Error', errorMessage);
  }
};

  // Función para descargar reportes
  const handleDownloadReports = () => {
    const dateStr = new Date().toISOString().split('T')[0];
    downloadFile('reportes/exportar-csv', `reportes_${dateStr}.csv`);
  };

  // Función para descargar accesos
  const handleDownloadAccess = () => {
    const dateStr = new Date().toISOString().split('T')[0];
    downloadFile('accesos/exportar-csv', `accesos_${dateStr}.csv`);
  };

  return (
    <View style={[styles.container, dynamicStyles.container]}>
      {/* Encabezado */}
      <View style={styles.header}>
        <Text style={[styles.title, dynamicStyles.text]}>BIOGATE</Text>
        <Text style={[styles.subtitle, dynamicStyles.subtext]}>
          Panel de administración
        </Text>
      </View>

      {/* Botones principales */}
      <View style={styles.buttonContainer}>
        <Pressable
          onPress={() => router.push('/horarios')}
          style={({ pressed }) => [
            styles.longButton,
            styles.blueButton,
            pressed && styles.pressedButton,
          ]}
        >
          <View style={styles.iconWithText}>
            <Ionicons name="calendar-outline" size={22} color="#fff" style={styles.icon} />
            <Text style={styles.cardText}>HORARIOS</Text>
          </View>
        </Pressable>

        <Pressable
          onPress={() => router.push('/elegir')}
          style={({ pressed }) => [
            styles.longButton,
            styles.blueButton,
            pressed && styles.pressedButton,
          ]}
        >
          <View style={styles.iconWithText}>
            <Ionicons name="document-text-outline" size={22} color="#fff" style={styles.icon} />
            <Text style={styles.cardText}>REPORTES</Text>
          </View>
        </Pressable>

        <Pressable
          onPress={() => router.push('/estadisticas')}
          style={({ pressed }) => [
            styles.longButton,
            styles.blueButton,
            pressed && styles.pressedButton,
          ]}
        >
          <View style={styles.iconWithText}>
            <Ionicons name="bar-chart-outline" size={22} color="#fff" style={styles.icon} />
            <Text style={styles.cardText}>ESTADÍSTICAS</Text>
          </View>
        </Pressable>
      </View>

      {/* Sección de descargas */}
      <View style={styles.downloadSection}>
        <Text style={[styles.downloadTitle, dynamicStyles.text]}>Descargar datos</Text>
        
        <Pressable
          onPress={handleDownloadReports}
          style={({ pressed }) => [
            styles.downloadButton,
            styles.greenButton,
            pressed && styles.pressedButton,
          ]}
        >
          <View style={styles.iconWithText}>
            <Ionicons name="download-outline" size={20} color="#fff" style={styles.icon} />
            <Text style={styles.downloadText}>Reportes (.csv)</Text>
          </View>
        </Pressable>

        <Pressable
          onPress={handleDownloadAccess}
          style={({ pressed }) => [
            styles.downloadButton,
            styles.purpleButton,
            pressed && styles.pressedButton,
          ]}
        >
          <View style={styles.iconWithText}>
            <Ionicons name="download-outline" size={20} color="#fff" style={styles.icon} />
            <Text style={styles.downloadText}>Accesos (.csv)</Text>
          </View>
        </Pressable>
      </View>

      {/* Barra inferior */}
      <BottomNavBar />
    </View>
  );
}

// Estilos (sin cambios)
const styles = StyleSheet.create({
  container: {
    flex: 1,
    paddingHorizontal: 24,
    paddingBottom: 100,
  },
  backButton: {
    position: 'absolute',
    top: 40,
    left: 20,
    padding: 10,
    borderRadius: 50,
    zIndex: 10,
  },
  header: {
    marginTop: 100,
    alignItems: 'center',
    marginBottom: 40,
  },
  title: {
    fontSize: 44,
    fontWeight: '900',
    letterSpacing: 1.8,
    textAlign: 'center',
  },
  subtitle: {
    fontSize: 16,
    marginTop: 4,
    textAlign: 'center',
  },
  buttonContainer: {
    flex: 1,
    justifyContent: 'center',
  },
  longButton: {
    width: '100%',
    height: 75,
    borderRadius: 24,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 24,
    elevation: 5,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.12,
    shadowRadius: 10,
  },
  iconWithText: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  icon: {
    marginRight: 6,
  },
  cardText: {
    fontSize: 18,
    fontWeight: '700',
    color: '#fff',
  },
  blueButton: {
    backgroundColor: '#005BFF',
  },
  greenButton: {
    backgroundColor: '#34C759',
  },
  purpleButton: {
    backgroundColor: '#AF52DE',
  },
  pressedButton: {
    opacity: 0.85,
  },
  downloadSection: {
    marginBottom: 40,
    paddingHorizontal: 16,
  },
  downloadTitle: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 12,
    textAlign: 'center',
  },
  downloadButton: {
    width: '100%',
    height: 50,
    borderRadius: 12,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 12,
    elevation: 3,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  downloadText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
  },
});

const lightStyles = StyleSheet.create({
  container: {
    backgroundColor: '#f8fafd',
  },
  backButton: {
    backgroundColor: '#e0e0e0',
  },
  text: {
    color: '#111',
  },
  subtext: {
    color: '#666',
  },
});

const darkStyles = StyleSheet.create({
  container: {
    backgroundColor: '#1C1C1E',
  },
  backButton: {
    backgroundColor: '#2a2a2c',
  },
  text: {
    color: '#fff',
  },
  subtext: {
    color: '#aaa',
  },
});