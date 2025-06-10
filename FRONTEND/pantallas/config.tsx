import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { useState } from 'react';
import {
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Switch,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { useTheme } from '../context/ThemeContext';
import BottomNavBar from './BottomNavBar';

export default function ConfiguracionScreen() {
  const router = useRouter();
  const { temaOscuro, toggleTema } = useTheme(); 
  const [notificaciones, setNotificaciones] = useState(true);

  const dynamicStyles = temaOscuro ? darkStyles : lightStyles;

  return (
    <SafeAreaView style={[styles.container, dynamicStyles.container]}>
      <ScrollView contentContainerStyle={styles.scroll}>
        {/* Header */}
        <View style={styles.header}>
          <Ionicons name="settings-outline" size={30} color={temaOscuro ? '#0A84FF' : '#007AFF'} />
          <Text style={[styles.headerText, dynamicStyles.headerText]}>Configuración</Text>
        </View>

        {/* Opciones */}
        <View style={[styles.section, dynamicStyles.section]}>
          <View style={styles.option}>
            <View style={styles.optionLeft}>
              <Ionicons name="moon" size={24} color={temaOscuro ? '#FFD60A' : '#5856D6'} />
              <Text style={[styles.optionText, dynamicStyles.optionText]}>Tema oscuro</Text>
            </View>
            <Switch
              value={temaOscuro}
              onValueChange={toggleTema} // usar función del contexto
              thumbColor="#fff"
              trackColor={{ false: '#ccc', true: '#0A84FF' }}
            />
          </View>

          <View style={styles.option}>
            <View style={styles.optionLeft}>
              <Ionicons name="notifications-outline" size={24} color={temaOscuro ? '#30D158' : '#FF9500'} />
              <Text style={[styles.optionText, dynamicStyles.optionText]}>Notificaciones</Text>
            </View>
            <Switch
              value={notificaciones}
              onValueChange={setNotificaciones}
              thumbColor="#fff"
              trackColor={{ false: '#ccc', true: '#30D158' }}
            />
          </View>
        </View>

        {/* Botón cerrar sesión */}
        <TouchableOpacity
          style={styles.logoutButton}
          activeOpacity={0.8}
          onPress={() => router.replace('/')}
        >
          <Ionicons name="log-out-outline" size={22} color="#fff" style={{ marginRight: 8 }} />
          <Text style={styles.logoutText}>Cerrar sesión</Text>
        </TouchableOpacity>

        {/* Términos y condiciones */}
        <View style={styles.termsContainer}>
          <Text style={[styles.termsTitle, dynamicStyles.optionText]}>
            Términos y Condiciones de Uso
          </Text>
          <Text style={[styles.termsText, dynamicStyles.optionText]}>
            Al utilizar esta aplicación de control biométrico, usted autoriza el procesamiento
            de su información personal para fines de acceso seguro. BioGate garantiza la protección
            de sus datos conforme a la legislación vigente en materia de privacidad. La información
            biométrica será utilizada únicamente para validar identidades en los accesos autorizados.
            Usted tiene derecho a solicitar la eliminación de sus datos previa petición formal.
          </Text>
          <Text style={[styles.termsText, dynamicStyles.optionText]}>
            Cualquier mal uso de la plataforma será sancionado conforme a nuestras políticas
            internas. BioGate no se hace responsable por accesos indebidos si las credenciales
            biométricas son comprometidas por el propio usuario.
          </Text>
        </View>
      </ScrollView>

      {/* BottomNavBar */}
      <BottomNavBar />
    </SafeAreaView>
  );
}


const styles = StyleSheet.create({
  container: {
    flex: 1,
    paddingTop: 50,
  },
  scroll: {
    paddingHorizontal: 20,
    paddingBottom: 90,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 30,
  },
  headerText: {
    fontSize: 24,
    fontWeight: '700',
    marginLeft: 10,
  },
  section: {
    borderRadius: 15,
    paddingVertical: 10,
    paddingHorizontal: 15,
    marginBottom: 30,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.08,
    shadowRadius: 8,
    elevation: 4,
  },
  option: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 14,
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  optionLeft: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  optionText: {
    fontSize: 16,
    marginLeft: 12,
  },
  logoutButton: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#FF3B30',
    borderRadius: 15,
    paddingVertical: 16,
    marginTop: 10,
    marginHorizontal: 30,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 5 },
    shadowOpacity: 0.15,
    shadowRadius: 10,
    elevation: 5,
  },
  logoutText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
  },
  termsContainer: {
    marginTop: 30,
    paddingHorizontal: 5,
  },
  termsTitle: {
    fontSize: 18,
    fontWeight: '700',
    marginBottom: 10,
  },
  termsText: {
    fontSize: 14,
    lineHeight: 20,
    marginBottom: 10,
    textAlign: 'justify',
  },
});

const lightStyles = StyleSheet.create({
  container: {
    backgroundColor: '#F9FAFB',
  },
  headerText: {
    color: '#222',
  },
  section: {
    backgroundColor: '#fff',
  },
  optionText: {
    color: '#333',
  },
});

const darkStyles = StyleSheet.create({
  container: {
    backgroundColor: '#1C1C1E',
  },
  headerText: {
    color: '#fff',
  },
  section: {
    backgroundColor: '#2C2C2E',
  },
  optionText: {
    color: '#f1f1f1',
  },
});
