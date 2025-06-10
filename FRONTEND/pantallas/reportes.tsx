import { Ionicons } from '@expo/vector-icons';
import { Picker } from '@react-native-picker/picker';
import { useRouter } from 'expo-router';
import { useState } from 'react';
import {
  Alert,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View
} from 'react-native';
import { useTheme } from '../context/ThemeContext';

type ApiError = {
  message?: string;
  detail?: string;
};

// Opciones para los combobox según la BD
const TIPOS_REPORTE = [
  { label: 'Error del sistema', value: 'Error del sistema' },
  { label: 'Fallo autenticación', value: 'Fallo autenticación' },
  { label: 'Fallo de dispositivo', value: 'Fallo de dispositivo' },
  { label: 'Acceso no autorizado', value: 'Acceso no autorizado' },
  { label: 'Horario irregular', value: 'Horario irregular' },
  { label: 'Otros', value: 'Otros' }
];

const NIVELES_SEVERIDAD = [
  { label: 'Baja', value: 'Baja' },
  { label: 'Media', value: 'Media' },
  { label: 'Alta', value: 'Alta' },
  { label: 'Crítica', value: 'Crítica' }
];

export default function EventosScreen() {
  const router = useRouter();
  const { temaOscuro } = useTheme();
  const dynamicStyles = temaOscuro ? darkStyles : lightStyles;

  const [titulo, setTitulo] = useState('Reporte de acceso denegado');
  const [descripcion, setDescripcion] = useState('');
  const [tipoReporte, setTipoReporte] = useState('Acceso no autorizado');
  const [severidad, setSeveridad] = useState('Media');
  const [idAccesoRelacionado, setIdAccesoRelacionado] = useState('');
  const [idDispositivo, setIdDispositivo] = useState('');
  const [evidencias, setEvidencias] = useState<string[]>([]);

  const goBack = () => {
    router.navigate('/elegir');
  };

   const handleSubmit = async () => {
    try {
      // Validar campos obligatorios
      if (!descripcion) {
        Alert.alert('Error', 'Por favor ingresa una descripción del reporte');
        return;
      }

      // Crear el objeto de reporte
      const reporteData = {
        titulo,
        descripcion,
        tipo_reporte: tipoReporte,
        severidad,
        id_acceso_relacionado: idAccesoRelacionado ? parseInt(idAccesoRelacionado) : null,
        id_dispositivo: idDispositivo ? parseInt(idDispositivo) : null,
        etiquetas: {},
        evidencias
      };

      // Enviar a la API
      const response = await fetch('https://render-biogate.onrender.com/reportes', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(reporteData)
      });

      if (!response.ok) {
        const errorData: ApiError = await response.json();
        throw new Error(errorData.detail || 'Error al crear el reporte');
      }

      const data = await response.json();
      
      Alert.alert('Éxito', 'Reporte creado correctamente', [
        { text: 'OK', onPress: () => router.replace('/home') }
      ]);
    } catch (error: unknown) {
      console.error('Error al crear reporte:', error);
      
      let errorMessage = 'No se pudo crear el reporte. Inténtalo de nuevo.';
      
      if (typeof error === 'object' && error !== null) {
        const apiError = error as ApiError;
        errorMessage = apiError.detail || apiError.message || errorMessage;
      } else if (typeof error === 'string') {
        errorMessage = error;
      }

      Alert.alert('Error', errorMessage);
    }
  };

  return (
    <SafeAreaView style={[{ flex: 1 }, dynamicStyles.container]}>
      <ScrollView contentContainerStyle={[styles.container, dynamicStyles.container]} showsVerticalScrollIndicator={false}>
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

        <Text style={[styles.header, dynamicStyles.text]}>Crear Reportes</Text>

        {/* Campos del formulario */}
        <View style={styles.field}>
          <Text style={[styles.label, dynamicStyles.subtext]}>Título del reporte:</Text>
          <TextInput
            style={[styles.editableInput, dynamicStyles.input]}
            placeholderTextColor={temaOscuro ? '#aaa' : '#999'}
            placeholder="Titulo descriptivo..."
          />
        </View>

        <View style={styles.field}>
          <Text style={[styles.label, dynamicStyles.subtext]}>Tipo de reporte:</Text>
          <View style={[styles.pickerContainer, dynamicStyles.input]}>
            <Picker
              selectedValue={tipoReporte}
              onValueChange={(itemValue) => setTipoReporte(itemValue)}
              style={[styles.picker, dynamicStyles.text]}
            >
              {TIPOS_REPORTE.map((item) => (
                <Picker.Item key={item.value} label={item.label} value={item.value} />
              ))}
            </Picker>
          </View>
        </View>

        <View style={styles.field}>
          <Text style={[styles.label, dynamicStyles.subtext]}>Severidad:</Text>
          <View style={[styles.pickerContainer, dynamicStyles.input]}>
            <Picker
              selectedValue={severidad}
              onValueChange={(itemValue) => setSeveridad(itemValue)}
              style={[styles.picker, dynamicStyles.text]}
            >
              {NIVELES_SEVERIDAD.map((item) => (
                <Picker.Item key={item.value} label={item.label} value={item.value} />
              ))}
            </Picker>
          </View>
        </View>

        <View style={styles.field}>
          <Text style={[styles.label, dynamicStyles.subtext]}>ID de acceso relacionado (opcional):</Text>
          <TextInput
            value={idAccesoRelacionado}
            onChangeText={setIdAccesoRelacionado}
            style={[styles.editableInput, dynamicStyles.input]}
            placeholder="Ej: 123"
            keyboardType="numeric"
            placeholderTextColor={temaOscuro ? '#aaa' : '#999'}
          />
        </View>

        <View style={styles.field}>
          <Text style={[styles.label, dynamicStyles.subtext]}>ID de dispositivo (opcional):</Text>
          <TextInput
            value={idDispositivo}
            onChangeText={setIdDispositivo}
            style={[styles.editableInput, dynamicStyles.input]}
            placeholder="Ej: 456"
            keyboardType="numeric"
            placeholderTextColor={temaOscuro ? '#aaa' : '#999'}
          />
        </View>

        {/* Descripción del reporte */}
        <Text style={[styles.label, dynamicStyles.subtext, { marginTop: 25 }]}>Descripción detallada:</Text>
        <TextInput
          style={[styles.input, dynamicStyles.input]}
          placeholder="Describe el problema o situación..."
          placeholderTextColor={temaOscuro ? '#aaa' : '#666'}
          multiline
          numberOfLines={4}
          value={descripcion}
          onChangeText={setDescripcion}
        />

        {/* Botón */}
        <TouchableOpacity style={styles.button} onPress={handleSubmit}>
          <Text style={styles.buttonText}>Crear Reporte</Text>
        </TouchableOpacity>
      </ScrollView>
    </SafeAreaView>
  );
}

// Estilos actualizados
const styles = StyleSheet.create({
  container: {
    paddingTop: 50,
    paddingHorizontal: 24,
    paddingBottom: 50,
  },
  header: {
    fontSize: 20,
    fontWeight: '600',
    textAlign: 'center',
    marginBottom: 25,
  },
  field: {
    marginBottom: 16,
  },
  label: {
    fontSize: 14,
    marginBottom: 4,
  },
  editableInput: {
    borderBottomWidth: 1,
    borderColor: '#ccc',
    paddingVertical: 6,
    fontSize: 15,
  },
  input: {
    borderRadius: 10,
    padding: 12,
    marginTop: 10,
    height: 100,
    textAlignVertical: 'top',
    fontSize: 15,
  },
  button: {
    backgroundColor: '#007AFF',
    borderRadius: 10,
    paddingVertical: 14,
    marginTop: 30,
    marginBottom: 20,
    alignItems: 'center',
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  closeButton: {
    position: 'absolute',
    top: 50,
    right: 24,
    width: 36,
    height: 36,
    borderRadius: 18,
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
  pickerContainer: {
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#ccc',
    marginTop: 5,
  },
  picker: {
    height: 60,
    width: '100%',
  },
});

// Estilos para modo claro
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
  input: {
    backgroundColor: '#F2F2F2',
    color: '#000',
  },
});

// Estilos para modo oscuro
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
  input: {
    backgroundColor: '#2C2C2E',
    color: '#fff',
  },
});