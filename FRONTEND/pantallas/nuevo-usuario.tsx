import { Ionicons } from '@expo/vector-icons';
import { Picker } from '@react-native-picker/picker';
import axios from 'axios';
import * as ImageManipulator from 'expo-image-manipulator';
import * as ImagePicker from 'expo-image-picker';
import { useRouter } from 'expo-router';
import { useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  Image,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import { useTheme } from '../context/ThemeContext';

type CountryCode = 'MX' | 'US' | 'ES' | 'AR' | 'CO';

interface PhoneFormat {
  codigo: string;
  formato: string;
}

type PhoneFormats = {
  [key in CountryCode]: PhoneFormat;
};

export default function NuevoUsuarioScreen() {
  const router = useRouter();
  const { temaOscuro } = useTheme(); 
  const dynamicStyles = temaOscuro ? darkStyles : lightStyles;
  
  const [aceptado, setAceptado] = useState(false);
  const [nombre, setNombre] = useState('');
  const [primerApellido, setPrimerApellido] = useState('');
  const [segundoApellido, setSegundoApellido] = useState('');
  const [telefono, setTelefono] = useState('');
  const [paisTelefono, setPaisTelefono] = useState<CountryCode>('MX');
  const [email, setEmail] = useState('');
  const [avatarImage, setAvatarImage] = useState<string | null>(null);
  const [facialImages, setFacialImages] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const formatosTelefono: PhoneFormats = {
    MX: { codigo: '+52', formato: 'XXX-XXX-XXXX' },
    US: { codigo: '+1', formato: 'XXX-XXX-XXXX' },
    ES: { codigo: '+34', formato: 'XXX XXX XXX' },
    AR: { codigo: '+54', formato: 'XX XXXX-XXXX' },
    CO: { codigo: '+57', formato: 'XXX XXX XXXX' }
  };

  // Solo para mostrar el avatar (no se usa en la API)
  const pickAvatarImage = async () => {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== 'granted') {
      alert('Se necesitan permisos para acceder a la galería');
      return;
    }

    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [1, 1],
      quality: 1,
    });

    if (!result.canceled && result.assets && result.assets.length > 0) {
      setAvatarImage(result.assets[0].uri);
    }
  };

  // Para seleccionar imágenes faciales (máximo 5)
  const pickFacialImages = async () => {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== 'granted') {
      alert('Se necesitan permisos para acceder a la galería');
      return;
    }

    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsMultipleSelection: true,
      selectionLimit: 10 - facialImages.length, // Cambiado de 5 a 10
      quality: 0.8,
    });

    if (!result.canceled && result.assets && result.assets.length > 0) {
      const resizedImages = await Promise.all(
        result.assets.map(async (asset) => {
          const resized = await ImageManipulator.manipulateAsync(
            asset.uri,
            [{ resize: { width: 500 } }],
            { compress: 0.8, format: ImageManipulator.SaveFormat.JPEG }
          );
          return resized.uri;
        })
      );

      setFacialImages([...facialImages, ...resizedImages].slice(0, 10)); // Cambiado de 5 a 10
    }
  };

  const removeFacialImage = (index: number) => {
    setFacialImages(facialImages.filter((_, i) => i !== index));
  };

  const registerUser = async () => {
    if (!nombre || !primerApellido) {
      Alert.alert('Error', 'Nombre y primer apellido son obligatorios');
      return;
    }

    if (facialImages.length === 0) {
      Alert.alert('Error', 'Debes seleccionar al menos una imagen facial');
      return;
    }

    setIsLoading(true);

    try {
      // Crear FormData para enviar a la API
      const formData = new FormData();

      // Agregar datos personales
      formData.append('nombre', nombre);
      formData.append('apellido_paterno', primerApellido);
      if (segundoApellido) formData.append('apellido_materno', segundoApellido);
      if (telefono) formData.append('telefono', formatosTelefono[paisTelefono].codigo + telefono);
      if (email) formData.append('correo_electronico', email);

      // Agregar imágenes como archivos
      facialImages.forEach((uri, index) => {
        const filename = uri.split('/').pop();
        const match = /\.(\w+)$/.exec(filename || '');
        const type = match ? `image/${match[1]}` : 'image/jpeg';

        formData.append('images', {
          uri,
          name: `facial_${index}.jpg`,
          type,
        } as any);
      });

      // Enviar a la API
      const response = await axios.post('https://render-biogate-990i.onrender.com/register_person', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (response.data && response.data.id_persona) {
        Alert.alert('Éxito', 'Usuario registrado correctamente');
        router.replace('/usuarios');
      } else {
        throw new Error('Respuesta inesperada del servidor');
      }
    } catch (error: any) {
      console.error('Error al registrar usuario:', error);
      let errorMessage = 'Error al registrar usuario';
      
      if (error.response) {
        if (error.response.data && error.response.data.detail) {
          errorMessage = error.response.data.detail;
        } else if (error.response.status === 400) {
          errorMessage = 'Datos inválidos enviados al servidor';
        } else if (error.response.status === 500) {
          errorMessage = 'Error interno del servidor';
        }
      }

      Alert.alert('Error', errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  const goBack = () => {
    router.navigate('/home');
  };

  return (
    <SafeAreaView style={[styles.container, dynamicStyles.container]}>
      <ScrollView contentContainerStyle={styles.scroll}>
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

        {/* Título */}
        <Text style={[styles.title, dynamicStyles.text]}>Nuevo usuario</Text>
        <Text style={[styles.subtitle, dynamicStyles.subtext]}>Registro del usuario</Text>

        {/* Avatar/Imagen (solo visual, no se usa en la API) */}
        <View style={styles.avatarContainer}>
          {avatarImage ? (
            <Image source={{ uri: avatarImage }} style={styles.avatarImage} />
          ) : (
            <Ionicons 
              name="person" 
              size={80} 
              color={temaOscuro ? '#ccc' : '#666'} 
            />
          )}
          <TouchableOpacity 
            style={[
              styles.imageButton,
              temaOscuro ? styles.imageButtonDark : styles.imageButtonLight
            ]} 
            onPress={pickAvatarImage}
          >
            <Text style={[
              styles.imageButtonText,
              temaOscuro ? styles.imageButtonTextDark : styles.imageButtonTextLight
            ]}>
              {avatarImage ? 'Cambiar imagen' : 'Seleccionar imagen'}
            </Text>
          </TouchableOpacity>
        </View>

        {/* Campos de nombre */}
        <Text style={[styles.label, dynamicStyles.subtext]}>Nombre(s)</Text>
        <TextInput 
          style={[styles.input, dynamicStyles.input]} 
          placeholder="Nombre(s)" 
          placeholderTextColor={temaOscuro ? '#aaa' : '#999'}
          value={nombre}
          onChangeText={setNombre}
        />

        <Text style={[styles.label, dynamicStyles.subtext]}>Primer apellido</Text>
        <TextInput 
          style={[styles.input, dynamicStyles.input]} 
          placeholder="Primer apellido" 
          placeholderTextColor={temaOscuro ? '#aaa' : '#999'}
          value={primerApellido}
          onChangeText={setPrimerApellido}
        />

        <Text style={[styles.label, dynamicStyles.subtext]}>Segundo apellido</Text>
        <TextInput 
          style={[styles.input, dynamicStyles.input]} 
          placeholder="Segundo apellido (opcional)" 
          placeholderTextColor={temaOscuro ? '#aaa' : '#999'}
          value={segundoApellido}
          onChangeText={setSegundoApellido}
        />

        {/* Teléfono con selector de país */}
        <Text style={[styles.label, dynamicStyles.subtext]}>Número de teléfono</Text>
        <View style={styles.phoneContainer}>
          <View style={[styles.countryPicker, dynamicStyles.input]}>
            <Picker
              selectedValue={paisTelefono}
              onValueChange={(itemValue) => setPaisTelefono(itemValue)}
              style={[dynamicStyles.text, { height: 50, width: 100 }]}
              dropdownIconColor={temaOscuro ? '#ccc' : '#555'}
            >
              <Picker.Item label="🇲🇽 +52" value="MX" />
              <Picker.Item label="🇺🇸 +1" value="US" />
              <Picker.Item label="🇪🇸 +34" value="ES" />
              <Picker.Item label="🇦🇷 +54" value="AR" />
              <Picker.Item label="🇨🇴 +57" value="CO" />
            </Picker>
          </View>
          <TextInput 
            style={[styles.input, dynamicStyles.input, styles.phoneInput]} 
            placeholder={formatosTelefono[paisTelefono].formato}
            keyboardType="phone-pad"
            placeholderTextColor={temaOscuro ? '#aaa' : '#999'}
            value={telefono}
            onChangeText={setTelefono}
          />
        </View>

        {/* Correo electrónico */}
        <Text style={[styles.label, dynamicStyles.subtext]}>Correo electrónico</Text>
        <TextInput 
          style={[styles.input, dynamicStyles.input]} 
          placeholder="ejemplo@dominio.com" 
          keyboardType="email-address"
          placeholderTextColor={temaOscuro ? '#aaa' : '#999'}
          value={email}
          onChangeText={setEmail}
        />

        {/* Sección de imágenes faciales */}
        <Text style={[styles.label, dynamicStyles.subtext]}>
          Imágenes faciales ({facialImages.length}/10)
        </Text>
        
        {/* Miniaturas de imágenes seleccionadas */}
        <View style={styles.thumbnailsContainer}>
          {facialImages.map((uri, index) => (
            <View key={index} style={styles.thumbnailWrapper}>
              <Image source={{ uri }} style={styles.thumbnail} />
              <TouchableOpacity 
                style={styles.removeThumbnailButton}
                onPress={() => removeFacialImage(index)}
              >
                <Ionicons name="close-circle" size={20} color="#ff3b30" />
              </TouchableOpacity>
            </View>
          ))}
        </View>

        {/* Botón para seleccionar imagen facial */}
        <TouchableOpacity 
          style={[
            styles.secondaryButton,
            temaOscuro ? styles.secondaryButtonDark : styles.secondaryButtonLight,
            facialImages.length >= 5 && styles.disabledButton
          ]}
          onPress={pickFacialImages}
          disabled={facialImages.length >= 5}
        >
          <Ionicons 
            name="image" 
            size={20} 
            color={temaOscuro ? '#fff' : '#007AFF'} 
            style={styles.buttonIcon}
          />
          <Text style={[
            styles.secondaryButtonText,
            temaOscuro ? styles.secondaryButtonTextDark : styles.secondaryButtonTextLight
          ]}>
            {facialImages.length >= 5 ? 
              'Máximo 10 imágenes' : 
              'Seleccionar imágenes de galería'}
          </Text>
        </TouchableOpacity>

        {/* Términos */}
        <View style={styles.checkboxContainer}>
          <TouchableOpacity onPress={() => setAceptado(!aceptado)} style={[styles.checkbox, dynamicStyles.checkbox]}>
            {aceptado && <View style={styles.checkboxChecked} />}
          </TouchableOpacity>
          <Text style={[styles.checkboxText, dynamicStyles.subtext]}>
            Acepto las <Text style={styles.link}>Condiciones del servicio</Text> y las{' '}
            <Text style={styles.link}>Políticas de privacidad de BioGate.</Text>
          </Text>
        </View>

        {/* Botón guardar */}
        <TouchableOpacity
          style={[styles.button, { opacity: aceptado ? 1 : 0.5 }]}
          disabled={!aceptado || isLoading}
          onPress={registerUser}
        >
          {isLoading ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text style={styles.buttonText}>Guardar</Text>
          )}
        </TouchableOpacity>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  scroll: {
    padding: 20,
    paddingBottom: 40,
  },
  title: {
    fontSize: 20,
    paddingTop: 40,
    fontWeight: '700',
    marginBottom: 4,
  },
  subtitle: {
    fontSize: 14,
    marginBottom: 25,
  },
  label: {
    fontSize: 14,
    marginBottom: 6,
  },
  input: {
    borderWidth: 1.5,
    borderRadius: 10,
    padding: 12,
    fontSize: 15,
    marginBottom: 15,
  },
  phoneContainer: {
    flexDirection: 'row',
    gap: 10,
    marginBottom: 15,
  },
  countryPicker: {
    borderWidth: 1.5,
    borderRadius: 10,
    justifyContent: 'center',
  },
  phoneInput: {
    flex: 1,
  },
  checkboxContainer: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 10,
    marginBottom: 25,
  },
  checkbox: {
    width: 20,
    height: 20,
    borderWidth: 1.5,
    borderRadius: 5,
    justifyContent: 'center',
    alignItems: 'center',
    marginTop: 3,
  },
  checkboxChecked: {
    width: 12,
    height: 12,
    backgroundColor: '#007AFF',
    borderRadius: 3,
  },
  checkboxText: {
    flex: 1,
    fontSize: 13,
  },
  link: {
    color: '#007AFF',
    textDecorationLine: 'underline',
  },
  button: {
    backgroundColor: '#007AFF',
    paddingVertical: 14,
    borderRadius: 10,
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
    right: 20,
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
  avatarContainer: {
    alignItems: 'center',
    marginBottom: 20,
  },
  avatarImage: {
    width: 120,
    height: 120,
    borderRadius: 60,
    marginBottom: 10,
  },
  imageButton: {
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderRadius: 20,
    marginTop: 10,
  },
  imageButtonLight: {
    backgroundColor: '#f0f0f0',
  },
  imageButtonDark: {
    backgroundColor: '#333',
  },
  imageButtonText: {
    fontSize: 14,
  },
  imageButtonTextLight: {
    color: '#007AFF',
  },
  imageButtonTextDark: {
    color: '#0a84ff',
  },
  secondaryButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 12,
    borderRadius: 10,
    marginBottom: 20,
    borderWidth: 1,
  },
  secondaryButtonLight: {
    backgroundColor: '#fff',
    borderColor: '#007AFF',
  },
  secondaryButtonDark: {
    backgroundColor: '#2c2c2e',
    borderColor: '#0a84ff',
  },
  secondaryButtonText: {
    fontSize: 16,
    marginLeft: 8,
  },
  secondaryButtonTextLight: {
    color: '#007AFF',
  },
  secondaryButtonTextDark: {
    color: '#0a84ff',
  },
  buttonIcon: {
    marginRight: 8,
  },
  disabledButton: {
    opacity: 0.5,
  },
  thumbnailsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
    marginBottom: 15,
  },
  thumbnailWrapper: {
    position: 'relative',
  },
  thumbnail: {
    width: 60,
    height: 60,
    borderRadius: 8,
  },
  removeThumbnailButton: {
    position: 'absolute',
    top: -8,
    right: -8,
    backgroundColor: 'white',
    borderRadius: 10,
  },
});

// ☀️ Estilos para modo claro
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
    borderColor: '#ccc',
    color: '#000',
  },
  checkbox: {
    borderColor: '#ccc',
  },
});

// 🌑 Estilos para modo oscuro
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
    borderColor: '#555',
    color: '#fff',
  },
  checkbox: {
    borderColor: '#555',
  },
});