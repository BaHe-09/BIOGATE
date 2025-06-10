import { useRouter } from 'expo-router';
import { useState } from 'react';
import { Alert, Image, StyleSheet, Text, TextInput, TouchableOpacity, View } from 'react-native';
import { useTheme } from '../context/ThemeContext';

export default function LoginScreen() {
  const router = useRouter();
  const { temaOscuro } = useTheme(); 
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const dynamicStyles = temaOscuro ? darkStyles : lightStyles;

  const handleRegisterPress = () => {
    router.push('/register');
  };

  const handleLoginPress = async () => {
  if (!username || !password) {
    Alert.alert('Error', 'Por favor ingresa usuario y contraseña');
    return;
  }

  setIsLoading(true);

  try {
    const response = await fetch('https://render-biogate.onrender.com/login/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        username: username,
        password: password,
      }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || 'Error de autenticación');
    }

    // Autenticación exitosa
    Alert.alert('Éxito', 'Inicio de sesión correcto');
    router.replace('/home');
    
  } catch (error) {
    let errorMessage = 'Error al conectar con el servidor';
    
    if (error instanceof Error) {
      errorMessage = error.message;
    }

    Alert.alert('Error', errorMessage);
  } finally {
    setIsLoading(false);
  }
};

  return (
    <View style={[styles.container, dynamicStyles.container]}>
      <Image source={require('../assets/images/biogate-logo.jpg')} style={styles.logo} />
      <TextInput
        style={[styles.input, dynamicStyles.input]}
        placeholder="Nombre de usuario"
        placeholderTextColor={temaOscuro ? '#ccc' : '#aaa'}
        value={username}
        onChangeText={setUsername}
        autoCapitalize="none"
      />
      <TextInput
        style={[styles.input, dynamicStyles.input]}
        placeholder="Contraseña"
        placeholderTextColor={temaOscuro ? '#ccc' : '#aaa'}
        value={password}
        onChangeText={setPassword}
        secureTextEntry
      />
      <TouchableOpacity 
        style={styles.button} 
        onPress={handleLoginPress}
        disabled={isLoading}
      >
        <Text style={styles.buttonText}>
          {isLoading ? 'Verificando...' : 'Iniciar sesión'}
        </Text>
      </TouchableOpacity>

      <View style={styles.register}>
        <View style={{ flexDirection: 'row' }}>
          <Text style={[styles.registerText, dynamicStyles.text]}>¿Aún no eres miembro? </Text>
          <TouchableOpacity onPress={handleRegisterPress}>
            <Text style={[styles.registerLink, dynamicStyles.link]}>Regístrate</Text>
          </TouchableOpacity>
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    paddingHorizontal: 30,
    justifyContent: 'center',
    alignItems: 'center',
  },
  logo: {
    width: 250,
    height: 250,
    resizeMode: 'contain',
    marginBottom: 70,
  },
  input: {
    width: '100%',
    height: 45,
    borderRadius: 10,
    paddingHorizontal: 15,
    marginBottom: 12,
    fontSize: 16,
  },
  button: {
    backgroundColor: '#007AFF',
    borderRadius: 10,
    paddingVertical: 12,
    width: '100%',
    alignItems: 'center',
    marginBottom: 10,
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
  },
  register: {
    marginTop: 10,
  },
  registerText: {
    fontSize: 14,
  },
  registerLink: {
    fontWeight: 'bold',
  },
});

const lightStyles = StyleSheet.create({
  container: {
    backgroundColor: '#fff',
  },
  text: {
    color: '#222',
  },
  input: {
    backgroundColor: '#F2F2F2',
    color: '#000',
  },
  link: {
    color: '#007AFF',
  },
});

const darkStyles = StyleSheet.create({
  container: {
    backgroundColor: '#1C1C1E',
  },
  text: {
    color: '#fff',
  },
  input: {
    backgroundColor: '#2C2C2E',
    color: '#fff',
  },
  link: {
    color: '#0A84FF',
  },
});
