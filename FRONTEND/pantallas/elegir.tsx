import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { SafeAreaView, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { useTheme } from '../context/ThemeContext';

export default function ElegirScreen() {
  const { temaOscuro } = useTheme();
  const router = useRouter();
  const dynamicStyles = temaOscuro ? darkStyles : lightStyles;

  return (
    <SafeAreaView style={[{ flex: 1 }, dynamicStyles.container]}>
      {/* Botón volver */}
      <TouchableOpacity
        onPress={() => router.replace('/administracion')}
        style={[styles.backButton, temaOscuro ? styles.backDark : styles.backLight]}
      >
        <Ionicons name="arrow-back" size={24} color={temaOscuro ? '#fff' : '#000'} />
      </TouchableOpacity>

      <View style={styles.body}>
        <Text style={[styles.header, dynamicStyles.text]}>Opciones de Reporte</Text>

        {/* Botón Crear Reporte */}
        <TouchableOpacity
          onPress={() => router.push('/reportes')}
          style={[styles.cardButton, styles.purpleButton]}
        >
          <Ionicons name="document-text-outline" size={28} color="#fff" style={styles.icon} />
          <Text style={styles.cardText}>Crear Reporte</Text>
        </TouchableOpacity>

        {/* Botón Ver Reportes */}
        <TouchableOpacity
          onPress={() => router.push('/registrorep')}
          style={[styles.cardButton, styles.greenButton]}
        >
          <Ionicons name="clipboard-outline" size={28} color="#fff" style={styles.icon} />
          <Text style={styles.cardText}>Ver Reportes</Text>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
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
    marginTop: 100,
    paddingHorizontal: 24,
    alignItems: 'center',
  },
  header: {
    fontSize: 24,
    fontWeight: '800',
    marginBottom: 40,
    textAlign: 'center',
  },
  cardButton: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 18,
    paddingHorizontal: 24,
    borderRadius: 20,
    marginBottom: 25,
    width: '100%',
    elevation: 6,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 5 },
    shadowOpacity: 0.12,
    shadowRadius: 8,
  },
  cardText: {
    fontSize: 18,
    fontWeight: '700',
    color: '#fff',
  },
  icon: {
    marginRight: 12,
  },
  purpleButton: {
    backgroundColor: '#6A5ACD',
  },
  greenButton: {
    backgroundColor: '#28A745',
  },
});

const lightStyles = StyleSheet.create({
  container: {
    backgroundColor: '#f2f6fc',
  },
  text: {
    color: '#111',
  },
});

const darkStyles = StyleSheet.create({
  container: {
    backgroundColor: '#1C1C1E',
  },
  text: {
    color: '#fff',
  },
});
