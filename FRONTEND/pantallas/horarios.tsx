import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { SafeAreaView, ScrollView, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { useTheme } from '../context/ThemeContext';

export default function HorariosScreen() {
  const { temaOscuro } = useTheme();
  const router = useRouter();
  const dynamicStyles = temaOscuro ? darkStyles : lightStyles;

  const volver = () => {
    router.replace('/administracion');
  };

  return (
    <SafeAreaView style={[{ flex: 1 }, dynamicStyles.container]}>
      <ScrollView contentContainerStyle={[styles.container]}>
        {/* Botón de volver */}
        <TouchableOpacity
          onPress={volver}
          style={[styles.backButton, temaOscuro ? styles.backDark : styles.backLight]}
        >
          <Ionicons name="arrow-back" size={24} color={temaOscuro ? '#fff' : '#000'} />
        </TouchableOpacity>

        {/* Título */}
        <Text style={[styles.header, dynamicStyles.text]}>Horarios de Acceso</Text>

        {/* Horario Matutino */}
        <View style={[styles.card, dynamicStyles.card]}>
          <Ionicons name="sunny" size={32} color={temaOscuro ? '#ffd479' : '#ffb300'} />
          <Text style={[styles.cardTitle, dynamicStyles.text]}>Matutino</Text>
          <Text style={[styles.cardTime, dynamicStyles.subtext]}>8:00 AM – 5:00 PM</Text>
        </View>

        {/* Horario Vespertino */}
        <View style={[styles.card, dynamicStyles.card]}>
          <Ionicons name="moon" size={32} color={temaOscuro ? '#9ecfff' : '#005eff'} />
          <Text style={[styles.cardTitle, dynamicStyles.text]}>Vespertino</Text>
          <Text style={[styles.cardTime, dynamicStyles.subtext]}>5:00 PM – 10:00 PM</Text>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    paddingTop: 60,
    paddingBottom: 60,
    paddingHorizontal: 24,
    alignItems: 'center',
  },
  header: {
    fontSize: 26,
    fontWeight: '700',
    marginBottom: 120,
    textAlign: 'center',
  },
  card: {
    width: '100%',
    alignItems: 'center',
    borderRadius: 20,
    padding: 24,
    marginBottom: 25,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 3 },
    shadowOpacity: 0.12,
    shadowRadius: 8,
    elevation: 4,
  },
  cardTitle: {
    fontSize: 20,
    fontWeight: '600',
    marginTop: 10,
  },
  cardTime: {
    fontSize: 16,
    marginTop: 4,
  },
  backButton: {
    position: 'absolute',
    top: 40,
    left: 20,
    zIndex: 10,
    padding: 10,
    borderRadius: 50,
  },
  backLight: {
    backgroundColor: 'rgba(0,0,0,0.05)',
  },
  backDark: {
    backgroundColor: 'rgba(255,255,255,0.1)',
  },
});

// Modo claro
const lightStyles = StyleSheet.create({
  container: {
    backgroundColor: '#ffffff',
  },
  text: {
    color: '#1a1a1a',
  },
  subtext: {
    color: '#555555',
  },
  card: {
    backgroundColor: '#f2f2f2',
  },
});

// Modo oscuro
const darkStyles = StyleSheet.create({
  container: {
    backgroundColor: '#1e1e1e',
  },
  text: {
    color: '#ffffff',
  },
  subtext: {
    color: '#cccccc',
  },
  card: {
    backgroundColor: '#2c2c2e',
  },
});
