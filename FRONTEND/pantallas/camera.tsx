import { MaterialCommunityIcons } from '@expo/vector-icons';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as ImageManipulator from 'expo-image-manipulator';
import * as ImagePicker from 'expo-image-picker';
import { useRouter } from 'expo-router';
import React, { useRef, useState } from 'react';
import { ActivityIndicator, Alert, Image, StyleSheet, Text, TouchableOpacity, View } from 'react-native';

const API_URL = 'https://biogate-detecom.onrender.com/get_embeddings';

interface FaceData {
  message: string;
  embedding_size: number;
  matches: Array<{
    id_persona: number;
    nombre_completo: string;
    similitud: number;
    id_vector: number;
    activo: boolean;
  }>;
  best_match?: {
    id_persona: number;
    nombre_completo: string;
    similitud: number;
    id_vector: number;
    activo: boolean;
  };
  access_granted: boolean;
  reason: string;
  estado_registro?: string; 
  hora_registro?: string;   
  dia_semana?: string;      
}

export default function FaceRecognitionScreen() {
  const router = useRouter();
  const [facing, setFacing] = useState<'front' | 'back'>('back');
  const [permission, requestPermission] = useCameraPermissions();
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [faceData, setFaceData] = useState<FaceData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const cameraRef = useRef<CameraView>(null);

  const handleSubmit = () => {
    router.replace('/home');
  };

  if (!permission) {
    return <View style={styles.container} />;
  }

  if (!permission.granted) {
    return (
      <View style={styles.permissionContainer}>
        <Text style={styles.permissionText}>Necesitamos acceso a la cámara</Text>
        <TouchableOpacity onPress={requestPermission} style={styles.permissionButton}>
          <Text style={styles.permissionButtonText}>Permitir cámara</Text>
        </TouchableOpacity>
        <TouchableOpacity onPress={handleSubmit} style={styles.closeButton}>
          <MaterialCommunityIcons name="close" size={24} color="black" />
        </TouchableOpacity>
      </View>
    );
  }

  const processAndResizeImage = async (uri: string) => {
    try {
      const manipResult = await ImageManipulator.manipulateAsync(
        uri,
        [{
          resize: { width: 500 } // Redimensiona manteniendo relación de aspecto
        }],
        { 
          compress: 0.9,
          format: ImageManipulator.SaveFormat.JPEG
        }
      );
      console.log(`Imagen redimensionada a: ${manipResult.width}x${manipResult.height}`);
      return manipResult.uri;
    } catch (error) {
      console.error('Error al procesar imagen:', error);
      throw new Error('Error al optimizar la imagen');
    }
  };

  const takePicture = async () => {
    if (!cameraRef.current) return;
    try {
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.8,
        skipProcessing: true,
        exif: false
      });

      const processedUri = await processAndResizeImage(photo.uri);
      setCapturedImage(processedUri);
      setFaceData(null);
      
    } catch (error) {
      console.error('Error al tomar foto:', error);
      Alert.alert('Error', 'No se pudo procesar la imagen');
    }
  };

  const pickImage = async () => {
    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        quality: 0.6
      });

      if (!result.canceled && result.assets?.[0]) {
        const processedUri = await processAndResizeImage(result.assets[0].uri);
        setCapturedImage(processedUri);
        setFaceData(null);
      }
    } catch (error) {
      console.error('Error al seleccionar imagen:', error);
      Alert.alert('Error', 'No se pudo cargar la imagen');
    }
  };

  const sendImage = async () => {
    if (!capturedImage) return;
    
    setIsLoading(true);
    try {
      const formData = new FormData();
      const file = {
        uri: capturedImage,
        name: 'photo.jpg',
        type: 'image/jpeg',
      };
      formData.append('file', file as any);

      const response = await fetch(API_URL, {
        method: 'POST',
        body: formData,
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'multipart/form-data',
        },
      });

      const responseText = await response.text();
      const data: FaceData = responseText ? JSON.parse(responseText) : {};

      if (!response.ok) {
        throw new Error(data.message || `Error ${response.status}`);
      }

      setFaceData({
        ...data,
        // Asegurar que los campos opcionales tengan valores por defecto
        access_granted: data.access_granted || false,
        reason: data.reason || 'No se pudo determinar la razón',
        matches: data.matches || [],
      });

      // Mostrar feedback basado en el acceso concedido
      if (data.access_granted) {
        Alert.alert('Acceso concedido', data.reason);
      } else {
        Alert.alert('Acceso denegado', data.reason);
      }
      
    } catch (error) {
      console.error('Error:', error);
      Alert.alert('Error', error instanceof Error ? error.message : 'Error al enviar imagen');
    } finally {
      setIsLoading(false);
    }
  };

  const toggleCamera = () => {
    setFacing(current => current === 'back' ? 'front' : 'back');
  };

  const retake = () => {
    setCapturedImage(null);
    setFaceData(null);
  };

  return (
    <View style={styles.container}>
      {!capturedImage ? (
        <View style={styles.cameraContainer}>
          <CameraView
            style={styles.camera}
            facing={facing}
            ref={cameraRef}
          />
          <TouchableOpacity onPress={handleSubmit} style={styles.topCloseButton}>
            <MaterialCommunityIcons name="close" size={32} color="white" />
          </TouchableOpacity>
          
          <View style={styles.controlsOverlay}>
            <TouchableOpacity onPress={toggleCamera} style={styles.flipButton}>
              <MaterialCommunityIcons name="camera-flip" size={28} color="white" />
            </TouchableOpacity>
            
            <TouchableOpacity onPress={takePicture} style={styles.captureButton}>
              <View style={styles.captureInner} />
            </TouchableOpacity>
            
            <TouchableOpacity onPress={pickImage} style={styles.galleryButton}>
              <MaterialCommunityIcons name="image" size={28} color="white" />
            </TouchableOpacity>
          </View>
        </View>
      ) : (
        <View style={styles.previewContainer}>
          <TouchableOpacity onPress={handleSubmit} style={styles.topCloseButton}>
            <MaterialCommunityIcons name="close" size={32} color="white" />
          </TouchableOpacity>
          
          <View style={styles.imageBox}>
  <Image source={{ uri: capturedImage }} style={styles.imageThumb} />
</View>
          
          {isLoading && (
            <View style={styles.loadingOverlay}>
              <ActivityIndicator size="large" color="#fff" />
              <Text style={styles.loadingText}>Procesando...</Text>
            </View>
          )}

          {faceData && (
            <View style={styles.resultsContainer}>
              <Text style={styles.resultTitle}>Resultados:</Text>
              
              {/* Estado de acceso */}
              <View style={[
                styles.accessStatus,
                faceData.access_granted ? styles.accessGranted : styles.accessDenied
              ]}>
                <Text style={styles.accessStatusText}>
                  {faceData.access_granted ? 'ACCESO CONCEDIDO' : 'ACCESO DENEGADO'}
                </Text>
              </View>
              
              {/* Estado de registro */}
              {faceData.estado_registro && (
                <View style={styles.registroStatus}>
                  <Text style={styles.registroStatusText}>
                    {faceData.estado_registro.toUpperCase()}
                  </Text>
                </View>
              )}
              
              {faceData.best_match && (
                <View style={styles.bestMatch}>
                  <Text style={styles.bestMatchTitle}>PERSONA DETECTADA</Text>
                  <Text style={styles.matchName}>{faceData.best_match.nombre_completo}</Text>
                  <Text style={styles.matchConfidence}>
                    Confianza: {(faceData.best_match.similitud * 100).toFixed(1)}%
                  </Text>
                  <Text style={styles.matchStatus}>
                    Estado: {faceData.best_match.activo ? 'ACTIVO' : 'INACTIVO'}
                  </Text>
                </View>
              )}
              
              {/* Información de fecha/hora */}
              {(faceData.hora_registro || faceData.dia_semana) && (
                <View style={styles.timeInfo}>
                  {faceData.hora_registro && (
                    <Text style={styles.timeText}>
                      Hora: {faceData.hora_registro}
                    </Text>
                  )}
                  {faceData.dia_semana && (
                    <Text style={styles.timeText}>
                      Día: {faceData.dia_semana}
                    </Text>
                  )}
                </View>
              )}
              
              <Text style={styles.reasonText}>Razón: {faceData.reason}</Text>
            </View>
          )}

          <View style={styles.buttonRow}>
            <TouchableOpacity onPress={retake} style={styles.actionButton}>
              <Text style={styles.buttonText}>Volver a tomar</Text>
            </TouchableOpacity>
            
            {!faceData && (
              <TouchableOpacity 
                onPress={sendImage} 
                style={[styles.actionButton, styles.sendButton]}
                disabled={isLoading}
              >
                {isLoading ? (
                  <ActivityIndicator color="white" />
                ) : (
                  <Text style={styles.buttonText}>Enviar</Text>
                )}
              </TouchableOpacity>
            )}
          </View>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: 'white',
  },
  permissionContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#fff',
    padding: 20,
  },
  permissionText: {
    fontSize: 16,
    marginBottom: 20,
  },
  permissionButton: {
    backgroundColor: '#007AFF',
    padding: 15,
    borderRadius: 10,
  },
  permissionButtonText: {
    color: 'white',
    fontSize: 16,
  },
  closeButton: {
    position: 'absolute',
    top: 40,
    right: 20,
  },
  cameraContainer: {
    flex: 1,
    position: 'relative',
  },
  camera: {
    flex: 1,
  },
  topCloseButton: {
    position: 'absolute',
    top: 40,
    left: 20,
    zIndex: 10,
    backgroundColor: 'rgba(0,0,0,0.5)',
    borderRadius: 20,
    padding: 5,
  },
  controlsOverlay: {
    position: 'absolute',
    bottom: 40,
    left: 0,
    right: 0,
    alignItems: 'center',
  },
  flipButton: {
    position: 'absolute',
    right: 20,
    top: 20,
    backgroundColor: '#007AFF',
    padding: 10,
    borderRadius: 30,
  },
  captureButton: {
    width: 70,
    height: 70,
    borderRadius: 35,
    backgroundColor: 'white',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 3,
    borderColor: 'rgba(255,255,255,0.5)',
  },
  captureInner: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: 'transparent',
    borderWidth: 2,
    borderColor: 'black',
  },
  galleryButton: {
    position: 'absolute',
    left: 20,
    bottom: 85,
    backgroundColor: '#007AFF',
    padding: 10,
    borderRadius: 30,
  },
  previewContainer: {
    flex: 1,
    backgroundColor: 'white',
    position: 'relative',
  },
  previewImage: {
    flex: 1,
  },
  loadingOverlay: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.7)',
  },
  loadingText: {
    color: 'white',
    marginTop: 15,
    fontSize: 16,
  },
  buttonRow: {
  position: 'absolute',
  bottom: 40, // ⬅Antes estaba en 40
  left: 20,
  right: 20,
  flexDirection: 'row',
  justifyContent: 'space-between',
},
  actionButton: {
    backgroundColor: '#007AFF',
    paddingVertical: 14,
    paddingHorizontal: 20,
    borderRadius: 12,
    minWidth: 150,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOpacity: 0.2,
    shadowRadius: 6,
    shadowOffset: { width: 0, height: 3 },
    elevation: 4,
  },
  sendButton: {
    backgroundColor: '#4CAF50',
  },
  buttonText: {
    color: 'white',
    fontWeight: 'bold',
    fontSize: 16,
  },
  resultsContainer: {
    position: 'absolute',
    bottom: 100,
    left: 20,
    right: 20,
    backgroundColor: 'rgba(0,0,0,0.7)',
    padding: 15,
    borderRadius: 10,
  },
  resultTitle: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 10,
    textAlign: 'center',
  },
  bestMatch: {
    marginBottom: 10,
  },
  bestMatchTitle: {
    color: '#4CAF50',
    fontSize: 14,
    fontWeight: 'bold',
    marginBottom: 5,
  },
  matchName: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
  matchConfidence: {
    color: '#FFC107',
    fontSize: 16,
  },
  matchStatus: {
    color: '#03A9F4',
    fontSize: 16,
  },
  accessStatus: {
    padding: 8,
    borderRadius: 5,
    marginBottom: 15,
    alignItems: 'center',
  },
  accessGranted: {
    backgroundColor: 'rgba(76, 175, 80, 0.3)',
    borderColor: '#4CAF50',
    borderWidth: 1,
  },
  accessDenied: {
    backgroundColor: 'rgba(244, 67, 54, 0.3)',
    borderColor: '#F44336',
    borderWidth: 1,
  },
  accessStatusText: {
    color: 'white',
    fontWeight: 'bold',
    fontSize: 16,
  },
  reasonText: {
    color: 'white',
    fontSize: 14,
    fontStyle: 'italic',
  },
  imageBox: {
  alignItems: 'center',
  marginTop: 180,   // ⬅bajamos un poco respecto al valor anterior de 40
  marginBottom: 30,
},
imageThumb: {
  width: 300,
  height: 300,
  borderRadius: 20,
  borderWidth: 3,
  borderColor: '#ffffff',
  resizeMode: 'cover',
},
registroStatus: {
    padding: 8,
    borderRadius: 5,
    marginBottom: 15,
    alignItems: 'center',
    backgroundColor: 'rgba(3, 169, 244, 0.3)',
    borderColor: '#03A9F4',
    borderWidth: 1,
  },
  registroStatusText: {
    color: 'white',
    fontWeight: 'bold',
    fontSize: 16,
  },
  timeInfo: {
    marginBottom: 10,
    padding: 8,
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: 5,
  },
  timeText: {
    color: 'white',
    fontSize: 14,
    marginBottom: 3,
  },

});