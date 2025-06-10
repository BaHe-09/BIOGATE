import { Picker } from '@react-native-picker/picker';
import { useRouter } from 'expo-router';
import { useState } from 'react';
import { Alert, Image, StyleSheet, Text, TextInput, TouchableOpacity, View } from 'react-native';
import { useTheme } from '../context/ThemeContext';

// Datos de países para el selector
const countries = [
  { code: '+52', name: 'México', flag: '🇲🇽' },
  { code: '+1', name: 'Estados Unidos', flag: '🇺🇸' },
  { code: '+34', name: 'España', flag: '🇪🇸' },
  { code: '+54', name: 'Argentina', flag: '🇦🇷' },
  { code: '+56', name: 'Chile', flag: '🇨🇱' },
  { code: '+57', name: 'Colombia', flag: '🇨🇴' },
];

type Country = typeof countries[0];

export default function RegisterScreen() {
  const router = useRouter();
  const { temaOscuro } = useTheme();
  const dynamicStyles = temaOscuro ? darkStyles : lightStyles;
  
  // Estados para los campos
  const [selectedCountry, setSelectedCountry] = useState<Country>(countries[0]);
  const [phone, setPhone] = useState('');
  const [name, setName] = useState('');
  const [lastName, setLastName] = useState('');
  const [secondLastName, setSecondLastName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleLoginPress = () => {
    router.push('/');
  };

  const validateFields = () => {
    if (!name || !lastName || !phone || !email || !password || !confirmPassword) {
      Alert.alert('Error', 'Por favor completa todos los campos obligatorios');
      return false;
    }

    if (password !== confirmPassword) {
      Alert.alert('Error', 'Las contraseñas no coinciden');
      return false;
    }

    if (password.length < 6) {
      Alert.alert('Error', 'La contraseña debe tener al menos 6 caracteres');
      return false;
    }

    // Validación simple de email
    if (!/^\S+@\S+\.\S+$/.test(email)) {
      Alert.alert('Error', 'Por favor ingresa un correo electrónico válido');
      return false;
    }

    return true;
  };

  const handleRegister = async () => {
  if (!validateFields()) return;

  setIsLoading(true);

  try {
    const userData = {
      persona: {
        name,
        lastName,
        secondLastName,
        phone: selectedCountry.code + phone,
        email
      },
      cuenta: {
        password,
        confirmPassword
      }
    };

    const response = await fetch('https://render-biogate.onrender.com/registrar/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(userData),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || 'Error en el registro');
    }

    Alert.alert(
      'Registro exitoso',
      'Tu cuenta ha sido creada correctamente',
      [
        { text: 'OK', onPress: () => router.push('/') }
      ]
    );
  } catch (error) {
    let errorMessage = 'Ocurrió un error al registrar. Por favor intenta nuevamente.';
    
    if (error instanceof Error) {
      errorMessage = error.message;
    } else if (typeof error === 'string') {
      errorMessage = error;
    }

    console.error('Error en registro:', error);
    Alert.alert('Error', errorMessage);
  } finally {
    setIsLoading(false);
  }
};

  return (
    <View style={[styles.container, dynamicStyles.container]}>
      <Image source={require('../assets/images/biogate-logo.jpg')} style={styles.logo} />

      <Text style={[styles.title, dynamicStyles.text]}>Crear Cuenta</Text>
      <Text style={[styles.subtitle, dynamicStyles.subtext]}>Regístrate para empezar</Text>

      {/* Campos de nombre completo */}
      <TextInput
        style={[styles.input, dynamicStyles.input]}
        placeholder="Nombre(s)*"
        placeholderTextColor={temaOscuro ? '#ccc' : '#aaa'}
        value={name}
        onChangeText={setName}
      />
      <TextInput
        style={[styles.input, dynamicStyles.input]}
        placeholder="Primer apellido*"
        placeholderTextColor={temaOscuro ? '#ccc' : '#aaa'}
        value={lastName}
        onChangeText={setLastName}
      />
      <TextInput
        style={[styles.input, dynamicStyles.input]}
        placeholder="Segundo apellido (opcional)"
        placeholderTextColor={temaOscuro ? '#ccc' : '#aaa'}
        value={secondLastName}
        onChangeText={setSecondLastName}
      />

      {/* Campo de teléfono con selector de país */}
      <View style={styles.phoneContainer}>
        <View style={[styles.countryPicker, dynamicStyles.input]}>
          <Picker
            selectedValue={selectedCountry.code}
            onValueChange={(itemValue: string) => {
              const country = countries.find(c => c.code === itemValue) || countries[0];
              setSelectedCountry(country);
            }}
            dropdownIconColor={temaOscuro ? '#ccc' : '#555'}
            mode="dropdown"
          >
            {countries.map((country) => (
              <Picker.Item 
                key={country.code} 
                label={`${country.flag} ${country.code}`} 
                value={country.code} 
              />
            ))}
          </Picker>
        </View>
        <TextInput
          style={[styles.phoneInput, dynamicStyles.input]}
          placeholder="XXX-XXX-XXXX*"
          placeholderTextColor={temaOscuro ? '#ccc' : '#aaa'}
          keyboardType="phone-pad"
          value={phone}
          onChangeText={setPhone}
        />
      </View>

      <TextInput
        style={[styles.input, dynamicStyles.input]}
        placeholder="Correo electrónico*"
        placeholderTextColor={temaOscuro ? '#ccc' : '#aaa'}
        keyboardType="email-address"
        autoCapitalize="none"
        value={email}
        onChangeText={setEmail}
      />
      <TextInput
        style={[styles.input, dynamicStyles.input]}
        placeholder="Contraseña*"
        placeholderTextColor={temaOscuro ? '#ccc' : '#aaa'}
        secureTextEntry
        value={password}
        onChangeText={setPassword}
      />
      <TextInput
        style={[styles.input, dynamicStyles.input]}
        placeholder="Confirmar contraseña*"
        placeholderTextColor={temaOscuro ? '#ccc' : '#aaa'}
        secureTextEntry
        value={confirmPassword}
        onChangeText={setConfirmPassword}
      />

      <TouchableOpacity 
        style={[styles.button, isLoading && styles.disabledButton]} 
        onPress={handleRegister}
        disabled={isLoading}
      >
        <Text style={styles.buttonText}>
          {isLoading ? 'Registrando...' : 'Registrarme'}
        </Text>
      </TouchableOpacity>

      <View style={styles.register}>
        <View style={{ flexDirection: 'row' }}>
          <Text style={[styles.registerText, dynamicStyles.subtext]}>¿Ya tienes cuenta? </Text>
          <TouchableOpacity onPress={handleLoginPress}>
            <Text style={[styles.registerLink, dynamicStyles.link]}>Inicia sesión</Text>
          </TouchableOpacity>
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    paddingHorizontal: 50,
    justifyContent: 'center',
    alignItems: 'center',
  },
  logo: {
    width: 120,
    height: 120,
    resizeMode: 'contain',
    marginBottom: 10,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    letterSpacing: 1,
    marginBottom: 5,
  },
  subtitle: {
    fontSize: 18,
    marginBottom: 20,
  },
  input: {
    width: '100%',
    height: 50,
    borderRadius: 10,
    paddingHorizontal: 15,
    marginBottom: 12,
    fontSize: 16,
  },
  phoneContainer: {
    flexDirection: 'row',
    width: '100%',
    marginBottom: 12,
  },
  countryPicker: {
    width: '30%',
    height: 50,
    borderRadius: 10,
    marginRight: 10,
    justifyContent: 'center',
  },
  picker: {
    width: '100%',
    height: '100%',
  },
  phoneInput: {
    flex: 1,
    height: 50,
    borderRadius: 10,
    paddingHorizontal: 15,
    fontSize: 16,
  },
  button: {
    backgroundColor: '#007AFF',
    borderRadius: 10,
    paddingVertical: 12,
    width: '100%',
    alignItems: 'center',
    marginTop: 10,
    marginBottom: 20,
  },
  disabledButton: {
    backgroundColor: '#007AFF80',
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
  },
  register: {
    marginTop: 5,
  },
  registerText: {
    fontSize: 14,
  },
  registerLink: {
    fontWeight: 'bold',
  },
});

// 🌞 Modo claro
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
  link: {
    color: '#007AFF',
  },
});

// 🌚 Modo oscuro
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
  link: {
    color: '#0A84FF',
  },
});